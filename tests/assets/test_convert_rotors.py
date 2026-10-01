"""``scripts/assets/convert.py`` keeps each rotor's ``NewtonActuator`` prim and authors the propeller
schema on it, so the converted vehicle declares its rotors the way the run reads them.
"""

import importlib.util
import pathlib

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

ROOT = pathlib.Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location("convert", ROOT / "scripts" / "assets" / "convert.py")
convert = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convert)

PARAMS = {"ct": 3.463e-6, "cd": 0.05, "rpm_max": 3800.0, "aero_h": 0.0, "aero_hforce": 0.0, "tau": 0.033}


def _source_vehicle(path):
    """A vehicle in the shape the converter reads: two rotor joints, each driven by a `NewtonActuator` prim."""
    stage = Usd.Stage.CreateNew(str(path))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Vehicle").GetPrim())
    UsdGeom.Xform.Define(stage, "/Vehicle/body")
    for i in range(2):
        UsdGeom.Xform.Define(stage, f"/Vehicle/rotor_{i}")
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/Vehicle/rotor_{i}_joint")
        joint.CreateBody0Rel().SetTargets(["/Vehicle/body"])
        joint.CreateBody1Rel().SetTargets([f"/Vehicle/rotor_{i}"])
        motor = stage.DefinePrim(f"/Vehicle/rotor_{i}_motor", "NewtonActuator")
        motor.CreateRelationship("newton:targets").SetTargets([joint.GetPath()])
        motor.CreateAttribute("newton:kd", Sdf.ValueTypeNames.Float).Set(0.025)
    stage.GetRootLayer().Save()
    return path


def test_the_converter_keeps_the_actuator_prims_and_authors_the_propeller_schema_on_them(tmp_path):
    """The asset converter keeps the `NewtonActuator` prims and authors the propeller schema on them.

    Given `scripts/assets/convert.py` run on a source vehicle, when it finishes, then each actuator prim
    applies the propeller schema and no joint authors `propeller:*`.
    """
    out = convert.author_rotor_params(_source_vehicle(tmp_path / "source.usda"), tmp_path / "out.usda", params=PARAMS)

    stage = Usd.Stage.Open(str(out))
    declared = {
        str(prim.GetPath()): "NexusPropellerAPI" in prim.GetAppliedSchemas()
        for prim in stage.Traverse()
        if prim.GetTypeName() == "NewtonActuator"
    }
    on_joints = [
        f"{prim.GetPath()}.{attr.GetName()}"
        for prim in stage.Traverse()
        if prim.GetTypeName() == "PhysicsRevoluteJoint"
        for attr in prim.GetAttributes()
        if attr.GetName().startswith("propeller:")
    ]
    assert (declared, on_joints) == ({"/Vehicle/rotor_0_motor": True, "/Vehicle/rotor_1_motor": True}, [])
