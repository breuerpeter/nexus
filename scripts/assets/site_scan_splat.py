"""Site Scan / SURE 3D-Tiles Gaussian-splat export → a single ``ParticleField3DGaussianSplat`` Universal Scene Description (USD).

Targets the layout an Esri Site Scan "Gaussian splat" download, from the SURE generator, ships: a 3D-Tiles
hierarchy, a root ``tileset.json`` referencing per-group tilesets, each a ``REPLACE``-refined quadtree
of ``.glb`` tiles across Level Of Detail (LOD) levels ``L0``, the finest, to ``L6``, the coarsest. Each glb carries
``KHR_gaussian_splatting`` + a gzip'd SPZ-2 buffer whose splat centers are in the tile's local frame;
the tile's placement, into Earth-Centered Earth-Fixed (ECEF) coordinates, lives in the accumulated
``transform`` chain down the tileset, so the script must apply those transforms; ignoring them stacks
every tile at the origin. Refinement is REPLACE, so a tile and its children cover the same ground at
different detail: the script walks the tileset and picks one tile per region at the requested detail,
``--max-geometric-error``, where 0 means leaves, the finest.

Pipeline: select tiles, with accumulated world transform → extract each SPZ → transform its splats to
ECEF → merge → rotate ECEF to East-North-Up (ENU) about the scene centroid, local Z-up; spherical, no
pyproj needed → recenter XY on the bbox center → ``write_gaussian_splat_usd`` → package as one ``.usdz``.
Where the drone starts isn't baked in: the registry scene's ``start``, suggested by
``scripts/assets/spawn_site.py``, places the chosen surface point at the world origin at load time.
The traced "scene origin" lat/lon/alt is the geodetic anchor of the scene frame: feed it to
``spawn_site.py --geo`` to get the registry ``geodetic_origin``, the geo of the start point.

Kit-only: it needs a booted Kit app, which ``nexus script`` provides::

    uv run nexus script scripts/assets/site_scan_splat.py \
        "$NEXUS_DATA/scans/gsp/Example Site" \
        --out "$NEXUS_DATA/scans/example_site_splat.usdz" --max-geometric-error 2.5
"""

from __future__ import annotations

import argparse


def convert_splat(tiles_dir, out_usdz, *, max_geometric_error=0.0) -> str:
    """3D-Tiles Gaussian-splat export → a recentred, Z-up ``ParticleField3DGaussianSplat`` ``.usdz``.

    Args:
        tiles_dir: the 3D-Tiles root, a dir with ``tileset.json``, a single ``.glb``, or a flat dir of
            ``.glb``, the legacy fallback, which warns; a flat dir carries no placement transforms.
        out_usdz: output ``.usdz`` path; scene assets are one self-contained file, the hosted
            convention and what the asset resolver caches.
        max_geometric_error: LOD selector; pick the coarsest tile whose geometricError ≤ this per
            region, under REPLACE refinement; ``0`` means descend to leaves, the finest and most tiles.
    Returns the output path.
    """
    import gzip
    import json
    import os
    import struct
    import tempfile

    import numpy as np
    from omni.kit.converter.gsplat import read_spz, write_gaussian_splat_usd
    from pxr import Sdf, UsdUtils

    out_usdz = str(out_usdz)
    if not out_usdz.endswith(".usdz"):
        raise ValueError(f"scene assets are ONE self-contained .usdz, got {out_usdz!r}")

    tiles = _collect_tiles(str(tiles_dir), float(max_geometric_error), np, json, os)
    if not tiles:
        raise FileNotFoundError(f"no .glb tiles found under {tiles_dir}")
    print(f"[site_scan_splat] selected {len(tiles)} tile(s) at max_geometric_error={max_geometric_error}")

    tmp = tempfile.mkdtemp(prefix="newton_splat_")
    fields = []
    for glb, world in tiles:
        field = read_spz(_glb_to_spz(glb, tmp, gzip, json, struct))
        _apply_world(field, world, np)  # local tile frame -> ECEF
        fields.append(field)

    field = fields[0] if len(fields) == 1 else _merge_splat_fields(fields, np)
    ref_geo = _ecef_to_enu(field, np)  # ECEF -> local ENU, Z up, about the scene centroid
    ctr = _recenter_xy(field, np)  # bbox-center XY -> origin, for precision; start placement stays registry data
    lat0, lon0 = _offset_geo(ref_geo, ctr, np)
    import sys

    print(
        f"[site_scan_splat] scene origin: lat {lat0:.6f}, lon {lon0:.6f}, alt {ref_geo[2]:.1f} m "
        f"(geo of the scene frame's origin/z=0; WGS84 ellipsoidal alt), pass to "
        f"`spawn_site.py --geo` to get the registry geodetic_origin at the picked start",
        file=sys.stderr,
    )

    conv_usd = os.path.join(tmp, "scene.usd")
    write_gaussian_splat_usd(field, conv_usd)
    # Stage-5 conventions: re-root the typed splat prim under one root Xform, the defaultPrim, and
    # author the sky: the scene USD is the single authority on its own lighting and start target.
    from scripts.assets.scene_root import finalize_scene_layer

    finalize_scene_layer(conv_usd)
    os.makedirs(os.path.dirname(out_usdz) or ".", exist_ok=True)
    if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(conv_usd), out_usdz):
        raise RuntimeError(f"usdz packaging failed for {conv_usd}")
    return out_usdz


# --- tileset walk: LOD selection plus accumulated placement transform ---


def _mat4(transform, np):
    """Turn a 3D-Tiles ``transform`` of 16 column-major floats into a 4x4 row-major matrix; identity if None."""
    if not transform:
        return np.eye(4)
    return np.array(transform, dtype=float).reshape(4, 4).T


def _collect_tiles(tiles_dir, max_gse, np, json, os):
    """Resolve the ``[(glb_path, world_4x4)]`` to convert. The walk of a ``tileset.json`` root
    applies LOD selection and transform accumulation, the correct path. A single ``.glb`` or flat dir
    goes in as-is with an identity transform, the legacy fallback, which warns.
    """
    if tiles_dir.endswith(".glb"):
        return [(tiles_dir, np.eye(4))]
    root_tileset = os.path.join(tiles_dir, "tileset.json")
    if os.path.isfile(root_tileset):
        out = []
        _walk_tileset(root_tileset, np.eye(4), max_gse, np, json, os, out)
        return out
    import glob as _glob

    glbs = sorted(_glob.glob(os.path.join(tiles_dir, "**", "*.glb"), recursive=True))
    if glbs:
        print(
            f"[site_scan_splat] WARNING: no tileset.json in {tiles_dir}, flat-merging {len(glbs)} glb "
            "with identity transforms (no placement / LOD selection)."
        )
    return [(g, np.eye(4)) for g in glbs]


def _walk_tileset(tileset_path, parent_world, max_gse, np, json, os, out):
    """Walk one ``tileset.json``, recursing into external ``.json`` tile refs, accumulating the world
    transform, and append ``(glb, world_4x4)`` for the tiles selected under REPLACE refinement: at
    each subtree take the coarsest tile whose ``geometricError`` ≤ *max_gse*, or a leaf if none is coarse
    enough, and stop; finer children cover the same ground.
    """
    base = os.path.dirname(tileset_path)
    with open(tileset_path) as fh:
        tileset = json.load(fh)

    def visit(node, parent):
        world = parent @ _mat4(node.get("transform"), np)
        content = node.get("content") or {}
        uri = content.get("uri") or content.get("url") or ""
        gse = float(node.get("geometricError", 0.0))
        children = node.get("children") or []
        if uri.endswith(".json"):  # external tileset: its root replaces this node
            _walk_tileset(os.path.join(base, uri), world, max_gse, np, json, os, out)
            return
        if uri.endswith(".glb") and (gse <= max_gse or not children):
            out.append((os.path.join(base, uri), world))
            return
        for child in children:
            visit(child, world)

    visit(tileset["root"], parent_world)


# --- geometry: place, orient, ground ---


def _apply_world(field, M, np):
    """Transform a tile's splats, positions and gaussian rotations, by its accumulated world matrix ``M``.
    The code assumes a rigid transform, rotation plus translation as in 3D-Tiles ECEF placement, and ignores scale.
    """
    R, T = M[:3, :3], M[:3, 3]
    field.positions = (R @ field.positions.T).T + T
    field.rotations = _quat_mul(_mat2quat(R, np), field.rotations, np)
    return field


def _ecef_to_enu(field, np):
    """Rotate the merged ECEF splats to a local ENU frame, Z the local up, about the scene centroid, so
    the scene is Z-up and near the origin. Spherical ENU, up the radial direction, which is exact enough
    for a scene spanning a few hundred metres and needs no pyproj. Returns the centroid's WGS84
    ``(lat, lon, alt)``: the geodetic anchor of the ENU frame, where z=0 is the centroid's ellipsoidal
    height.
    """
    p = field.positions
    ref = p.mean(axis=0)
    geo = _ecef_to_wgs84(ref, np)
    up = ref / np.linalg.norm(ref)
    east = np.cross(np.array([0.0, 0.0, 1.0]), up)
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    R = np.stack([east, north, up])  # rows = ENU basis: R @ (p - ref) gives ENU coords
    field.positions = (R @ (p - ref).T).T
    field.rotations = _quat_mul(_mat2quat(R, np), field.rotations, np)
    return geo


def _ecef_to_wgs84(p, np):
    """ECEF in m → WGS84 ``(lat°, lon°, ellipsoidal alt m)`` via Bowring's closed form, with no pyproj."""
    a, f = 6378137.0, 1.0 / 298.257223563
    b = a * (1.0 - f)
    e2, ep2 = 1.0 - (b / a) ** 2, (a / b) ** 2 - 1.0
    x, y, z = p
    lon = np.arctan2(y, x)
    r = np.hypot(x, y)
    theta = np.arctan2(z * a, r * b)
    lat = np.arctan2(z + ep2 * b * np.sin(theta) ** 3, r - e2 * a * np.cos(theta) ** 3)
    alt = r / np.cos(lat) - a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    return float(np.degrees(lat)), float(np.degrees(lon)), float(alt)


def _recenter_xy(field, np):
    """Recenter XY on the bbox center; z stays the centroid-relative ENU height. Where the drone
    starts is deliberately not baked into the asset: the registry scene's ``start``, suggested by
    ``scripts/assets/spawn_site.py``, places the chosen surface point at the world origin at load
    time. Returns the applied ``(cx, cy)`` shift, the scene origin's ENU offset from the centroid.
    """
    p = field.positions
    cx = (p[:, 0].min() + p[:, 0].max()) / 2.0
    cy = (p[:, 1].min() + p[:, 1].max()) / 2.0
    field.positions = p - np.array([cx, cy, 0.0])
    return float(cx), float(cy)


def _offset_geo(geo, enu_xy, np):
    """Geodetic ``(lat, lon)`` of the point ``(east, north)`` metres from ``geo``: a spherical
    small-offset shift, sub-centimetre over a few hundred metres.
    """
    r_earth = 6371008.8
    lat, lon, _ = geo
    x, y = enu_xy
    return lat + float(np.degrees(y / r_earth)), lon + float(np.degrees(x / (r_earth * np.cos(np.radians(lat)))))


def _mat2quat(R, np):
    """Rotation matrix (3x3) → quaternion (w, x, y, z)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return np.array([w, x, y, z])


def _quat_mul(q, batch, np):
    """Left-multiply every quaternion in ``batch`` (N,4, wxyz) by the single quaternion ``q`` (wxyz)."""
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    )


def _merge_splat_fields(fields, np):
    """Concatenate many ``GaussianSplatData`` tiles into the first, in place. The dataclass fields
    are positions/scales/rotations/opacities/f_dc, all required, plus optional f_rest, the higher-order SH;
    ``count``, which the writer uses, needs a refresh after concatenation.
    """
    base = fields[0]
    for attr in ("positions", "scales", "rotations", "opacities", "f_dc"):
        setattr(base, attr, np.concatenate([getattr(f, attr) for f in fields], axis=0))
    if all(getattr(f, "f_rest", None) is not None for f in fields):
        base.f_rest = np.concatenate([f.f_rest for f in fields], axis=0)
    else:
        base.f_rest = None
    base.count = base.positions.shape[0]
    return base


def _glb_to_spz(glb_path: str, out_dir: str, gzip, json, struct) -> str:
    """Extract bufferView 0 of a ``KHR_gaussian_splatting`` glb, raw SPZ with the "NGSP" magic, to a
    gzip'd ``.spz``, since gsplat's read_spz reads a gzip stream.
    """
    import os

    with open(glb_path, "rb") as fh:
        data = fh.read()
    if data[:4] != b"glTF":
        raise ValueError(f"not a glb: {glb_path}")
    json_len = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20 : 20 + json_len].decode("utf-8"))
    bin_start = 20 + json_len + 8  # skip the binary chunk header, length plus type
    bv = gltf["bufferViews"][0]
    off = bin_start + bv.get("byteOffset", 0)
    blob = data[off : off + bv["byteLength"]]
    if blob[:2] != b"\x1f\x8b":  # raw SPZ -> gzip it for read_spz
        blob = gzip.compress(blob)
    out = os.path.join(out_dir, os.path.splitext(os.path.basename(glb_path))[0] + ".spz")
    with open(out, "wb") as fh:
        fh.write(blob)
    return out


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser, factored out so it can be parse-tested without Kit."""
    p = argparse.ArgumentParser(
        prog="site_scan_splat", description="Site Scan / SURE 3D-Tiles Gaussian splat -> ParticleField USD."
    )
    p.add_argument("src", help="3D-Tiles root (dir with tileset.json), a .glb, or a flat dir of .glb")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--max-geometric-error", type=float, default=0.0,
        help="LOD budget: pick the coarsest tile with geometricError <= this per region (0 = finest/leaves)",
    )  # fmt: skip
    return p


def main(argv=None) -> None:
    import sys

    def _t(m):  # Kit swallows stdout; trace on stderr
        print(f"[site_scan_splat] {m}", file=sys.stderr, flush=True)

    args = build_parser().parse_args(argv)
    _t(f"src={args.src!r} out={args.out!r} max_gse={args.max_geometric_error}")
    try:
        import omni.kit.app

        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
            "omni.kit.converter.gsplat", True
        )
        _t("gsplat enabled; converting…")
        out = convert_splat(args.src, args.out, max_geometric_error=args.max_geometric_error)
        _t(f"DONE -> {out}")
        print(out, flush=True)
    except Exception:
        import traceback

        _t("EXCEPTION:\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
