"""``scripts/assets/convert.py`` declares the camera it authors with the camera schema, not with ``sensor:`` attributes."""

import importlib.util
import pathlib

from pxr import Usd, UsdGeom, UsdPhysics

import nexus_sim  # noqa: F401  # registers the nexus schema plugin

ROOT = pathlib.Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location("convert", ROOT / "scripts" / "assets" / "convert.py")
convert = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convert)


def test_the_asset_converter_authors_the_sensor_schemas(tmp_path):
    """The asset converter authors the sensor schemas.

    Given `scripts/assets/convert.py` run on a source vehicle of one rigid body, when it has authored the
    camera, then the camera prim applies the camera schema, with the sensor role it includes, and no prim
    authors a `sensor:` attribute.
    """
    source = tmp_path / "source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Vehicle").GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/Vehicle/body").GetPrim())
    stage.GetRootLayer().Save()

    out = convert.author_vehicle_camera(str(source), str(tmp_path / "out.usda"))

    authored = Usd.Stage.Open(out)
    schemas = list(authored.GetPrimAtPath("/Vehicle/body/FpvCam").GetAppliedSchemas())
    legacy = [
        str(a.GetPath()) for p in authored.Traverse() for a in p.GetAttributes() if a.GetName().startswith("sensor:")
    ]
    assert (schemas, legacy) == (["NexusCameraAPI", "NexusSensorRoleAPI"], [])
