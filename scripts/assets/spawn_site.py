"""Suggest a ``start``, the drone's start point, for a converted First Person View (FPV) scene
Universal Scene Description (USD).

Placement is per-scene registry data, not baked into the asset: the registry scene's
``start: [x, y, z]`` is the scene-frame point the runtime places at the world origin, so the
drone, created at the origin and resting on the z=0 physics ground, starts on that surface.
This tool suggests one. It's advisory: eyeball the result, tweak the numbers in
``registry.yaml`` if the scene calls for a different spot; either way, no reconversion.

Host-runnable with usd-core + numpy and no Kit::

    uv run python scripts/assets/spawn_site.py <scene.usdz> [--geo lat,lon[,alt]]

``--geo`` is the geodetic anchor of the scene frame's origin, which the splat converter traces as
"scene origin: …"; with it, the tool also prints the registry ``geodetic_origin``: the
lat/lon plus ellipsoidal alt of the picked start point, which is the world origin the runtime
places it at.

The suggestion is a flat, open-terrain cell near the scene center:

1. rasterize the scene's points, mesh vertices or splat centers, into an XY height map with
   square cells; a cell's surface height is a high percentile of its z values, robust to
   stray splats / photogrammetry flyers;
2. keep cells inside a central window, half the bbox, since scans are water or fall-off at the
   fringe, fully surrounded in the 3x3 neighborhood and locally flat, spread < ``FLAT_SPREAD``;
3. cluster the flat cells' heights and take the lowest mode with enough area,
   ``MIN_MODE_CELLS``: the open-terrain level, rejecting large flat rooftops higher up and
   stray flat outliers lower down;
4. suggest the cell of that mode nearest the scene center.

Falls back to the surface under the bbox-center column when no flat mode exists: small, steep,
or sparsely sampled scenes.
"""

from __future__ import annotations

import argparse

CELL = 4.0  # m: height-map cell size
MIN_PTS = 8  # points a cell needs to carry a height
SURFACE_PCT = 95  # per-cell surface = this z percentile; rejects flyers, keeps the roof/ground
FLAT_SPREAD = 1.5  # m: max 3x3 neighborhood height spread for a cell to count as flat
BIN = 2.0  # m: height-mode histogram bin
MIN_MODE_CELLS = 8  # cells, ~128 m^2, a height mode needs to count as open terrain
ELIG_TOL = 1.5  # m: flat cells within this of the mode height are start candidates


def pick_spawn_site(positions):
    """The suggested start site for a scene.

    Args:
        positions: (N, 3) world-space point positions, mesh vertices or splat centers, metres,
            Z up.

    Returns:
        ``(x, y, z, how)``: the site in scene coordinates, where ``z`` is the surface height there,
        and a human-readable description of the pick.
    """
    import numpy as np

    p = np.asarray(positions)
    mn, mx = p.min(axis=0), p.max(axis=0)
    cx, cy = float(mn[0] + mx[0]) / 2.0, float(mn[1] + mx[1]) / 2.0

    # Height map: per-cell surface height, the SURFACE_PCT percentile of the cell's z.
    ij = np.floor((p[:, :2] - np.array([cx, cy])) / CELL).astype(np.int64)
    keys, inv = np.unique(ij, axis=0, return_inverse=True)
    counts = np.bincount(inv)
    z_by_cell = np.split(p[np.argsort(inv, kind="stable"), 2], np.cumsum(counts)[:-1])
    heights = {
        (int(k[0]), int(k[1])): float(np.percentile(zs, SURFACE_PCT))
        for k, zs in zip(keys, z_by_cell, strict=True)
        if len(zs) >= MIN_PTS
    }

    def spread(i, j):
        hs = [heights.get((i + di, j + dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1)]
        return None if any(h is None for h in hs) else max(hs) - min(hs)

    wx, wy = float(mx[0] - mn[0]) / 4.0, float(mx[1] - mn[1]) / 4.0
    flat = {
        (i, j): h
        for (i, j), h in heights.items()
        if abs((i + 0.5) * CELL) < wx
        and abs((j + 0.5) * CELL) < wy
        and (s := spread(i, j)) is not None
        and s < FLAT_SPREAD
    }

    if flat:
        hs = np.sort(np.array(list(flat.values())))
        nbins = max(1, int(np.ceil((hs[-1] - hs[0]) / BIN)))
        hist, edges = np.histogram(hs, bins=nbins)
        for count, lo, hi in zip(hist, edges[:-1], edges[1:], strict=True):  # ascending: lowest mode first
            if count < MIN_MODE_CELLS:
                continue
            mode = float(np.median(hs[(hs >= lo) & (hs <= hi)]))
            elig = {k: h for k, h in flat.items() if abs(h - mode) <= ELIG_TOL}
            i, j = min(elig, key=lambda k: k[0] ** 2 + k[1] ** 2)
            x, y = cx + (i + 0.5) * CELL, cy + (j + 0.5) * CELL
            how = (
                f"open terrain at height {mode:+.1f} m ({count} flat cells), "
                f"{((x - cx) ** 2 + (y - cy) ** 2) ** 0.5:.0f} m from the scene centre"
            )
            return x, y, elig[(i, j)], how

    # Fallback: the surface under the bbox-center column; widen once over a center gap, then take
    # the lowest point in the whole scene.
    for radius in (8.0, 48.0):
        near = (np.abs(p[:, 0] - cx) < radius) & (np.abs(p[:, 1] - cy) < radius)
        if near.any():
            return cx, cy, float(p[near, 2].max()), f"fallback: bbox-centre column max (r={radius:.0f} m)"
    return cx, cy, float(p[:, 2].min()), "fallback: global minimum"


def scene_points(usd_path: str):
    """All world-space points in the USD: mesh/points prims plus Gaussian-splat fields. Their
    ``ParticleField3DGaussianSplat`` schema is unknown to plain usd-core, so read the ``points``/
    ``positions`` attribute generically and apply the prim's local-to-world transform.
    """
    import numpy as np
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise FileNotFoundError(usd_path)
    xf = UsdGeom.XformCache()
    chunks = []
    for prim in stage.Traverse():
        attr = prim.GetAttribute("points") or prim.GetAttribute("positions")
        v = attr.Get() if attr else None
        if not v:
            continue
        m = np.array(xf.GetLocalToWorldTransform(prim))  # row-major, row-vector convention
        pts = np.asarray(v, dtype=np.float64)
        chunks.append(pts @ m[:3, :3] + m[3, :3])
    if not chunks:
        raise ValueError(f"no point-bearing prims found in {usd_path}")
    return np.concatenate(chunks)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spawn_site", description=__doc__.splitlines()[0])
    p.add_argument("usd", help="converted scene USD (photogrammetry mesh or Gaussian splat)")
    p.add_argument(
        "--geo",
        default=None,
        metavar="LAT,LON[,ALT]",
        help="geodetic anchor of the scene frame's origin (the converter's 'scene origin' trace);"
        " prints the registry geodetic_origin AT the picked start",
    )
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    x, y, z, how = pick_spawn_site(scene_points(args.usd))
    print(f"pick: {how}")
    print(f"registry scene entry:  start: [{x:.1f}, {y:.1f}, {z:.1f}]")
    if args.geo:
        import math

        parts = [float(v) for v in args.geo.split(",")]
        lat0, lon0, alt0 = parts[0], parts[1], (parts[2] if len(parts) > 2 else None)
        r_earth = 6371008.8
        lat = lat0 + math.degrees(y / r_earth)
        lon = lon0 + math.degrees(x / (r_earth * math.cos(math.radians(lat0))))
        alt = f", alt: {alt0 + z:.1f}" if alt0 is not None else ""
        print(f"                       geodetic_origin: {{ lat: {lat:.6f}, lon: {lon:.6f}{alt} }}")


if __name__ == "__main__":
    main()
