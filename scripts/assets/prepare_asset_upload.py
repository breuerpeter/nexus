"""Prepare a vehicle or scene Universal Scene Description (USD) for **manual** upload to
content-addressed asset storage.

This does everything *except* upload; you upload by hand, out of band:

1. sha256 the USD: the content address and the registry version pin.
2. Compute the flat content-addressed URL to upload it to, under the catalog's own base:

       <base>/assets/usd/vehicles/<name>-<sha256>.usdz     (a --vehicle asset)
       <base>/assets/usd/scenes/<name>-<sha256>.usdz       (a --scene asset)

   ``<name>`` is the USD's filename stem. Hosted assets are self-contained ``.usdz``, one blob and
   one sha256, so the hash sits in the key and nothing ever overwrites the object: a
   Content Delivery Network (CDN) caches it forever. ``--base`` publishes somewhere else, for
   example a bucket of your own; the registry snippet then carries the full URL rather than the
   compact ref, which only the catalog's own base completes.
3. For a **vehicle** only: convert it to a web-ready ``.glb`` preview, the ``<model-viewer>``
   companion, written next to the USD under ``assets/local/`` so the docs preview works before
   you publish the asset; ``docs/hooks/vehicle_previews.py`` serves that local copy when present,
   and this writes nothing into ``docs/``. Its upload key is the ``.glb`` sibling of the USD key.
   **Scenes get no glb.**
4. Print the files to upload, their upload keys + CloudFront URLs, and the `registry.yaml`
   snippet to paste in.

Usage, run in the project env so the sha helper imports:

    uv run python scripts/assets/prepare_asset_upload.py --vehicle assets/local/astro_max_fpv.usdz [--rotate-x 180]
    uv run python scripts/assets/prepare_asset_upload.py --scene   assets/local/slalom.usdz

The glb conversion needs Blender-as-a-module, `bpy`, whose wheels pin a specific
Python minor, cp311, that differs from the project interpreter, so this script re-invokes
*itself* for that step in an isolated interpreter: `uv run --no-project --python 3.11 --with
bpy --with usd-core`. That keeps the whole asset-prep flow in one file with no duplicated
conversion code.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_ASSETS = ROOT / "assets" / "local"  # preview-before-publish glb copies live beside the USDs

# ──────────────────────────────────────────────────────────────────────────────
# glb conversion: runs in the isolated `bpy` interpreter; see the module docstring.
# Everything under here imports bpy/pxr lazily so the orchestrator env, which has no bpy, can
# import this module without pulling Blender in.
# ──────────────────────────────────────────────────────────────────────────────


def _apply_usd_material_colors(usd_path: Path) -> None:
    """Set Blender material colors from the USD, keyed by material name.

    The colors live on each Material prim's interface inputs, since the PreviewSurface's
    diffuseColor is *connected* to a `material_N.diffuseColor` parameter, which Blender's
    importer leaves at default gray. Read them with pxr and write Base Color / Metallic /
    Roughness onto the matching Principled Bidirectional Scattering Distribution Function (BSDF)
    node. If pxr isn't installed, skip with a warning.
    """
    import bpy

    try:
        from pxr import Usd, UsdShade
    except ImportError:
        print("warning: usd-core (pxr) unavailable; skipping material recolor", file=sys.stderr)
        return

    stage = Usd.Stage.Open(str(usd_path))
    usd_mats: dict[str, dict] = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            mat = UsdShade.Material(prim)
            usd_mats[prim.GetName()] = {
                inp.GetBaseName(): inp.Get() for inp in mat.GetInputs() if inp.Get() is not None
            }

    for bmat in bpy.data.materials:
        props = usd_mats.get(bmat.name) or usd_mats.get(bmat.name.split(".")[0])
        if not props or not bmat.use_nodes:
            continue
        bsdf = next((n for n in bmat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue
        dc = props.get("diffuseColor")
        if dc is not None:
            bsdf.inputs["Base Color"].default_value = (dc[0], dc[1], dc[2], 1.0)
        for usd_key, bsdf_slot in (("metallic", "Metallic"), ("roughness", "Roughness")):
            if usd_key in props and bsdf_slot in bsdf.inputs:
                bsdf.inputs[bsdf_slot].default_value = float(props[usd_key])


def _rotate_scene_x(degrees: float) -> None:
    """Rotate every root object about world X, for assets that author a flipped 'up'."""
    import bpy
    from mathutils import Matrix

    rot = Matrix.Rotation(math.radians(degrees), 4, "X")
    for obj in bpy.context.scene.objects:
        if obj.parent is None:
            obj.matrix_world = rot @ obj.matrix_world


def _export_glb(src: Path, dst: Path, rotate_x: float) -> None:
    import bpy

    # Start from an empty scene so the default cube/camera/light never leak in.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Blender's USD importer reads .usd/.usda/.usdc/.usdz; usdz is a zip of the others.
    bpy.ops.wm.usd_import(filepath=str(src))
    _apply_usd_material_colors(src)
    if rotate_x:
        _rotate_scene_x(rotate_x)
    bpy.ops.export_scene.gltf(
        filepath=str(dst),
        export_format="GLB",  # single self-contained binary: geometry + textures
        export_yup=True,  # glTF is Y-up; converts from USD's authored up-axis
        export_image_format="WEBP",  # compress embedded textures; model-viewer decodes it
        export_image_quality=85,
    )


def _optimize(raw: Path, dst: Path, texture_size: int) -> bool:
    """Web-optimize ``raw`` into ``dst`` with gltf-transform. Returns False if npx is missing."""
    npx = shutil.which("npx")
    if npx is None:
        return False
    cmd = [npx, "--yes", "@gltf-transform/cli", "optimize", str(raw), str(dst),
           "--texture-compress", "webp", "--compress", "draco"]  # fmt: skip
    if texture_size:
        # Active only when gltf-transform's `sharp` backend is present; harmless otherwise.
        cmd += ["--texture-size", str(texture_size)]
    subprocess.run(cmd, check=True)
    return True


def _convert(src: Path, dst: Path, *, texture_size: int, rotate_x: float) -> None:
    """USD → web-ready glb: bpy export, then a gltf-transform optimize pass when npx is present."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.glb"
        _export_glb(src, raw, rotate_x)
        if not _optimize(raw, dst, texture_size):
            print("warning: npx/gltf-transform not found; writing unoptimized glb", file=sys.stderr)
            shutil.copyfile(raw, dst)


def _convert_main(argv: list[str]) -> int:
    """Entry point for the isolated `bpy` re-invocation: `... _convert <src> <dst> [--rotate-x D]`."""
    p = argparse.ArgumentParser(prog="prepare_asset_upload.py _convert")
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--texture-size", type=int, default=1024)
    p.add_argument("--rotate-x", type=float, default=0.0)
    a = p.parse_args(argv)
    _convert(a.src, a.dst, texture_size=a.texture_size, rotate_x=a.rotate_x)
    print(f"wrote {a.dst} ({a.dst.stat().st_size / 1e6:.1f} MB)")
    return 0


def _reinvoke_convert(src: Path, dst: Path, rotate_x: float) -> None:
    """Run the glb conversion in an isolated py3.11 + bpy interpreter: this file, in `_convert` mode."""
    cmd = ["uv", "run", "--no-project", "--python", "3.11", "--with", "bpy", "--with", "usd-core",
           "python", str(Path(__file__)), "_convert", str(src), str(dst)]  # fmt: skip
    if rotate_x:
        cmd += ["--rotate-x", str(rotate_x)]
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError(f"conversion produced no glb at {dst}")


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator: runs in the project env.
# ──────────────────────────────────────────────────────────────────────────────


def _catalog_base() -> str:
    """The base the catalog this repo ships names, so a printed URL matches what a run would fetch."""
    from nexus._src.config import load_registry

    return load_registry().assets.base or ""


def _registry_snippet(kind: str, name: str, sha: str, url: str, compact: bool) -> str:
    # An asset under the catalog's own base is compact, because that base completes it. One published
    # anywhere else carries its full URL, which is how a catalog names a blob hosted elsewhere.
    ref = f"{{ name: {name}, sha256: {sha} }}" if compact else f'{{ url: "{url}", sha256: {sha} }}'
    usd = f"    usd: {ref}"
    if kind == "vehicle":
        return f"  - name: {name}\n{usd}\n    px4: {{ airframe: TODO }}   # assign per payload"
    return f"  {name}:\n{usd}\n    geodetic_origin: null   # or {{ lat: .., lon: .. }} for a geo-anchored scene"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument("--vehicle", type=Path, metavar="USD", help="prepare a vehicle asset (USD + glb preview)")
    kind.add_argument("--scene", type=Path, metavar="USD", help="prepare a scene asset (USD only, no glb)")
    parser.add_argument(
        "--rotate-x", type=float, default=0.0, metavar="DEG",
        help="glb: rotate about world X (180 for assets authored 'up = -Z', e.g. Astro Max)",
    )  # fmt: skip
    parser.add_argument(
        "--base", default=None, metavar="URL",
        help="publish under this base instead of the catalog's own, for example a bucket of your own",
    )  # fmt: skip
    args = parser.parse_args(argv)

    is_vehicle = args.vehicle is not None
    usd: Path = args.vehicle if is_vehicle else args.scene
    if not usd.exists():
        parser.error(f"input not found: {usd}")
    if usd.suffix != ".usdz":
        parser.error(f"hosted assets must be self-contained .usdz (got '{usd.suffix}'): package it as .usdz first")

    from nexus._src.assets.resolver import hosted_url, sha256_file  # reuse the scheme + hasher, no duplicate

    kind_dir = "vehicles" if is_vehicle else "scenes"
    name = usd.stem  # <name> is the filename stem; the registry stores just this + the sha
    sha = sha256_file(usd)
    catalog_base = _catalog_base()
    base = args.base or catalog_base
    if not base:
        parser.error("this catalog declares no assets.base: pass --base <url> to say where to publish")
    compact = base.rstrip("/") == catalog_base.rstrip("/")  # only the catalog's own base completes a compact ref
    usd_url = hosted_url(base, f"usd/{kind_dir}", name, sha)

    print(f"kind      {kind_dir[:-1]}")
    print(f"name      {name}")
    print(f"sha256    {sha}")
    print("\nupload to:")
    print(f"  {usd}  ->  {usd_url}")

    if is_vehicle:
        glb_url = hosted_url(base, f"usd/{kind_dir}", name, sha, "glb")
        glb_local = LOCAL_ASSETS / f"{name}-{sha}.glb"
        print(f"  {glb_local}  ->  {glb_url}")
        print("\nconverting to glb (isolated bpy env) ...")
        _reinvoke_convert(usd, glb_local, args.rotate_x)
        print(f"  glb  -> {glb_url}  (local preview copy: {glb_local.relative_to(ROOT)})")
        if not compact:
            print("  NOTE: the docs preview reads the local copy until this base is one the site can reach.")

    print(f"\nregistry.yaml snippet (add under `{kind_dir}:`):\n")
    print(_registry_snippet("vehicle" if is_vehicle else "scene", name, sha, usd_url, compact))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_convert":
        raise SystemExit(_convert_main(sys.argv[2:]))
    raise SystemExit(main(sys.argv[1:]))
