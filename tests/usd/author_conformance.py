"""Author the conformance fixture: a minimal vehicle Universal Scene Description (USD) file that applies every schema the nexus plugin defines.

Run `uv run python tests/usd/author_conformance.py` to rewrite `conformance.usda` beside this script, or
pass an output path. A test checks that the committed file matches this script's output byte for byte.
"""

import sys
from pathlib import Path

from pxr import Gf, Usd, UsdGeom

import nexus_sim  # noqa: F401  # registers the nexus schema plugin


def author(out: Path) -> None:
    """Write the fixture to `out`: PX4 and its Software In The Loop (SITL) peer on the default prim, one body under it with an Inertial Measurement Unit (IMU) on a lever arm and one prim for each other sensor schema, and one rotor body with a propeller."""
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    vehicle = UsdGeom.Xform.Define(stage, "/Vehicle")
    stage.SetDefaultPrim(vehicle.GetPrim())
    vehicle.GetPrim().ApplyAPI("NexusPx4API")
    vehicle.GetPrim().GetAttribute("nexus:airframe").Set("astro_max")
    vehicle.GetPrim().ApplyAPI("NexusPx4SitlAPI")
    UsdGeom.Xform.Define(stage, "/Vehicle/body")
    imu = UsdGeom.Xform.Define(stage, "/Vehicle/body/Imu")
    imu.AddTranslateOp().Set(Gf.Vec3d(0.01, -0.02, 0.03))
    prim = imu.GetPrim()
    prim.ApplyAPI("NexusImuAPI")
    prim.GetAttribute("nexus:accNoise").Set(0.05)
    for name, schema in (("Mag", "NexusMagAPI"), ("Baro", "NexusBaroAPI"), ("Gps", "NexusGpsAPI")):
        UsdGeom.Xform.Define(stage, f"/Vehicle/body/{name}").GetPrim().ApplyAPI(schema)
    UsdGeom.Camera.Define(stage, "/Vehicle/body/Cam").GetPrim().ApplyAPI("NexusCameraAPI")
    UsdGeom.Camera.Define(stage, "/Vehicle/body/Ir").GetPrim().ApplyAPI("NexusThermalCameraAPI")
    # OpenUSD doesn't know Kit's lidar type here, so it can't check the schema against it.
    stage.DefinePrim("/Vehicle/body/Lidar", "OmniLidar").AddAppliedSchema("NexusLidarAPI")
    rotor = UsdGeom.Xform.Define(stage, "/Vehicle/rotor").GetPrim()
    rotor.ApplyAPI("NexusPropellerAPI")
    rotor.GetAttribute("nexus:ct").Set(3.463e-6)
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    author(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("conformance.usda"))
