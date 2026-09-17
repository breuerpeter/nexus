"""Photogrammetry OBJ → render-only self-contained ``.usdz``, site-agnostic.

Converts a single-OBJ photogrammetry mesh, a directory holding a ``.obj`` plus ``.mtl`` and textures,
for example an Esri Site Scan "OBJ" export, but any single-OBJ dir works, into one self-contained
``.usdz``, geometry and textures packaged together, via ``omni.kit.asset_converter`` +
``UsdUtils.CreateNewUsdzPackage``. Scene assets are single-file by convention: the asset resolver
caches exactly one file, so sibling texture dirs would go missing. This route isn't
Site-Scan-specific: it just takes the first ``.obj`` it finds.

Kit-only: it needs a booted Kit app, which ``nexus script`` provides::

    uv run nexus script scripts/assets/obj_to_usd.py \
        "$NEXUS_DATA/scans/obj/Example Site" --out "$NEXUS_DATA/scans/example_site_obj.usdz"
"""

from __future__ import annotations

import argparse


def convert_obj(obj_dir, out_usdz, *, recenter="spawn") -> str:
    """Photogrammetry OBJ → one self-contained render-only ``.usdz``.

    Recipe: recenter the OBJ XY on the bbox center, since source projected coordinates are ~400 km out
    → float precision loss, and keep z as authored; run ``omni.kit.asset_converter`` with
    ``export_preview_surface=True``, ``smooth_normals=True``, ``convert_stage_up_z=True``,
    ``use_meter_as_world_unit=True`` into a scratch dir, where the converter emits a ``.usd`` + a
    sibling ``textures/`` dir; then package layer + textures into the single output ``.usdz``.
    Kit-only; runs operationally in the container.

    Args:
        obj_dir: a directory containing the ``.obj`` plus ``.mtl`` and textures, or the ``.obj`` path.
        out_usdz: output ``.usdz`` path; scene assets are single-file, see the module docstring.
        recenter: ``"spawn"``, the default, recenters per the recipe; ``"none"`` keeps source coords.
    Returns the output path.
    """
    import asyncio
    import glob
    import os
    import shutil
    import tempfile

    import omni.kit.asset_converter as kit_ac
    from pxr import Sdf, UsdUtils

    out_usdz = str(out_usdz)
    if not out_usdz.endswith(".usdz"):
        raise ValueError(f"scene assets are ONE self-contained .usdz, got {out_usdz!r}")

    obj_dir = str(obj_dir)
    if os.path.isdir(obj_dir):
        objs = sorted(glob.glob(os.path.join(obj_dir, "*.obj")))
        if not objs:
            raise FileNotFoundError(f"no .obj found in {obj_dir}")
        obj_path = objs[0]
    else:
        obj_path = obj_dir

    src = obj_path
    tmp = None
    if recenter == "spawn":
        tmp = tempfile.mkdtemp(prefix="newton_mesh_")
        src = _recenter_obj(obj_path, tmp)

    ctx = kit_ac.AssetConverterContext()  # 6.0.1 API; the old create_converter_context() no longer exists
    ctx.export_preview_surface = True
    ctx.smooth_normals = True
    ctx.convert_stage_up_z = True
    ctx.use_meter_as_world_unit = True

    conv_dir = tempfile.mkdtemp(prefix="newton_mesh_usd_")
    conv_usd = os.path.join(conv_dir, "scene.usd")
    task = kit_ac.get_instance().create_converter_task(src, conv_usd, None, ctx)
    # The converter is an async Kit task: it only progresses while something pumps the app update loop,
    # so the code must schedule the coroutine and tick app.update() until it resolves; a bare
    # run_until_complete never advances Kit and returns without converting.
    import omni.kit.app

    app = omni.kit.app.get_app()
    fut = asyncio.ensure_future(task.wait_until_finished())
    while not fut.done():
        app.update()
    ok = fut.result()
    if not ok:
        raise RuntimeError(f"asset_converter failed for {src}: {task.get_error_message()}")

    # Stage-5 conventions: one root prim, the defaultPrim with no root xformOps, + the authored sky:
    # the scene Universal Scene Description (USD) file is the single authority on its own lighting and
    # start placement target.
    from scripts.assets.scene_root import finalize_scene_layer

    finalize_scene_layer(conv_usd)
    os.makedirs(os.path.dirname(out_usdz) or ".", exist_ok=True)
    if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(conv_usd), out_usdz):
        raise RuntimeError(f"usdz packaging failed for {conv_usd}")
    shutil.rmtree(conv_dir, ignore_errors=True)
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    return out_usdz


def _recenter_obj(obj_path: str, out_dir: str) -> str:
    """Copy an OBJ plus siblings into ``out_dir`` with vertices recentered XY on the bbox center,
    for float precision, z kept as authored. Where the drone starts is deliberately not baked into
    the asset: the registry scene's ``start``, suggested by ``scripts/assets/spawn_site.py``,
    places the chosen surface point at the world origin at load time. The copy keeps the sidecar
    files, ``.mtl`` and textures, verbatim. Returns the recentered ``.obj`` path.
    """
    import os
    import shutil
    import sys

    src_dir = os.path.dirname(obj_path) or "."
    for f in os.listdir(src_dir):
        s = os.path.join(src_dir, f)
        if os.path.isfile(s) and not f.endswith(".obj"):
            shutil.copy2(s, os.path.join(out_dir, f))

    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    with open(obj_path) as fh:
        for line in fh:
            if line.startswith("v "):
                _, xs, ys, _zs = line.split()[:4]
                x, y = float(xs), float(ys)
                minx, maxx = min(minx, x), max(maxx, x)
                miny, maxy = min(miny, y), max(maxy, y)
    ox, oy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    print(f"[obj_to_usd] recentred XY on the bbox centre ({ox:.1f}, {oy:.1f}); z as authored", file=sys.stderr)

    out_obj = os.path.join(out_dir, os.path.basename(obj_path))
    with open(obj_path) as fh, open(out_obj, "w") as out:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                x, y, z = float(parts[1]) - ox, float(parts[2]) - oy, float(parts[3])
                out.write(f"v {x} {y} {z}" + ("".join(" " + p for p in parts[4:]) if len(parts) > 4 else "") + "\n")
            else:
                out.write(line)
    return out_obj


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser, factored out so it can be parse-tested without Kit."""
    p = argparse.ArgumentParser(prog="obj_to_usd", description="Photogrammetry OBJ -> render-only USD.")
    p.add_argument("src", help="OBJ directory (or .obj file)")
    p.add_argument("--out", required=True, help="output .usdz path (ONE self-contained file)")
    p.add_argument("--recenter", default="spawn", choices=["spawn", "none"])
    return p


def main(argv=None) -> None:
    import sys

    def _t(m):  # Kit swallows stdout; trace on stderr so progress is visible
        print(f"[obj_to_usd] {m}", file=sys.stderr, flush=True)

    args = build_parser().parse_args(argv)
    _t(f"src={args.src!r} out={args.out!r} recenter={args.recenter}")
    try:
        import omni.kit.app

        # asset_converter isn't in every experience's autoload set: enable it defensively.
        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.kit.asset_converter", True)
        _t("asset_converter enabled; converting…")
        out = convert_obj(args.src, args.out, recenter=args.recenter)
        _t(f"DONE -> {out}")
        print(out, flush=True)
    except Exception:
        import traceback

        _t("EXCEPTION:\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
