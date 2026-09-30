"""Author the conformance fixture: a minimal vehicle Universal Scene Description (USD) file that applies every schema the nexus plugin defines.

Run `uv run python tests/usd/author_conformance.py` to rewrite `conformance.usda` beside this script, or
pass an output path. A test checks that the committed file matches this script's output byte for byte.
"""

import sys
from pathlib import Path

from pxr import Gf, Usd, UsdGeom

import nexus  # noqa: F401  # registers the nexus schema plugin


def author(out: Path) -> None:
    """Write the fixture to `out`: PX4 on the default prim, and one body under it with an Inertial Measurement Unit (IMU) on a lever arm."""
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    vehicle = UsdGeom.Xform.Define(stage, "/Vehicle")
    stage.SetDefaultPrim(vehicle.GetPrim())
    vehicle.GetPrim().ApplyAPI("NexusPx4API")
    vehicle.GetPrim().GetAttribute("nexus:airframe").Set("astro_max")
    UsdGeom.Xform.Define(stage, "/Vehicle/body")
    imu = UsdGeom.Xform.Define(stage, "/Vehicle/body/Imu")
    imu.AddTranslateOp().Set(Gf.Vec3d(0.01, -0.02, 0.03))
    prim = imu.GetPrim()
    prim.ApplyAPI("NexusImuAPI")
    prim.GetAttribute("nexus:accNoise").Set(0.05)
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    author(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("conformance.usda"))
