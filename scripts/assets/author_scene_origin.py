"""Author where a scene sits on Earth into a scene package: its ``PhysicsScene`` prim and ``nexus:geodeticOrigin``.

A run fails on a scene whose ``PhysicsScene`` authors no origin. This adds the prim, at
``/World/PhysicsScene`` with ``NexusSceneAPI``, to a scene ``.usdz`` that has none, such as a converted
scan, and writes a new package beside it, every other file unchanged:

    uv run python scripts/assets/author_scene_origin.py <scene.usdz> <out.usdz> --origin LAT,LON,ALT

``LAT`` and ``LON`` are degrees, and ``ALT`` is the altitude over mean sea level in metres, or for a
streamed globe the surface's WGS84 ellipsoidal height. ``spawn_site.py --geo`` gives a scan's origin at
its start. Then prepare the package for upload with ``prepare_asset_upload.py --scene``.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # `scene_root` beside this file, run from any folder

from scene_root import author_origin  # noqa: E402


def author(scene: Path, out: Path, origin: tuple[float, float, float]) -> None:
    """Write `out`, the package `scene` with a ``PhysicsScene`` that authors `origin`.

    Raises:
        ValueError: `scene` already holds a ``PhysicsScene`` prim.
        RuntimeError: Packaging failed.
    """
    from pxr import Sdf, Usd, UsdUtils

    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(scene) as package:
        package.extractall(tmp)
        root = Path(tmp) / package.namelist()[0]  # a package's first file is its root layer
        stage = Usd.Stage.Open(str(root))
        held = [str(p.GetPath()) for p in stage.Traverse() if p.GetTypeName() == "PhysicsScene"]
        if held:
            raise ValueError(f"{scene}: already holds {', '.join(held)}; edit its nexus:geodeticOrigin instead")
        author_origin(stage, origin)
        stage.GetRootLayer().Save()
        if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(root)), str(out)):
            raise RuntimeError(f"usdz packaging failed for {root}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", type=Path, help="the scene .usdz to read")
    parser.add_argument("out", type=Path, help="the .usdz to write")
    parser.add_argument("--origin", required=True, metavar="LAT,LON,ALT", help="degrees, degrees, metres")
    args = parser.parse_args(argv)
    lat, lon, alt = (float(v) for v in args.origin.split(","))
    author(args.scene, args.out, (lat, lon, alt))
    print(f"wrote {args.out}: /World/PhysicsScene at {lat}, {lon}, {alt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
