"""``scripts/assets/convert.py`` authors the propeller schema on each rotor's rigid body and leaves each
``NewtonActuator`` prim with Newton's schemas alone, so the converted vehicle declares its rotors the
way the run reads them. A rotor is a joint the source declares as one, or one the caller names: a
joint Newton drives for another purpose, such as a gimbal servo, is none.
"""

import importlib.util
import pathlib

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

ROOT = pathlib.Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location("convert", ROOT / "scripts" / "assets" / "convert.py")
convert = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convert)

PARAMS = {"ct": 3.463e-6, "cd": 0.05, "aero_h": 0.0, "aero_hforce": 0.0, "tau": 0.033}
NEWTON = ["NewtonPIDControlAPI", "NewtonDCMotorClampingAPI"]


def _source_vehicle(path, *, declared=True, limit=397.935):
    """A vehicle in the shape of the pinned ones: two rotor bodies and a gimbal on revolute joints, each joint
    driven by a `NewtonActuator` prim with Newton's schemas. The gimbal is no rotor.

    Args:
        path: Where to write the vehicle.
        declared: Whether the two rotor joints author `propeller:*`, the declaration the schema replaces.
        limit: The no-load speed each motor declares [rad/s]; 397.935 is the joints' 3800 rpm.
    """
    stage = Usd.Stage.CreateNew(str(path))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Vehicle").GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/Vehicle/body").GetPrim())
    for name in ("rotor_0", "rotor_1", "gimbal"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, f"/Vehicle/{name}").GetPrim())
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/Vehicle/{name}_joint")
        joint.CreateBody0Rel().SetTargets(["/Vehicle/body"])
        joint.CreateBody1Rel().SetTargets([f"/Vehicle/{name}"])
        if declared and name != "gimbal":
            for attr, value in {"propeller:ct": 3.463e-6, "propeller:cd": 0.05, "propeller:rpm_max": 3800.0}.items():
                joint.GetPrim().CreateAttribute(attr, Sdf.ValueTypeNames.Float, custom=True).Set(value)
        motor = stage.DefinePrim(f"/Vehicle/{name}_motor", "NewtonActuator")
        motor.CreateRelationship("newton:targets").SetTargets([joint.GetPath()])
        for schema in NEWTON:
            motor.ApplyAPI(schema)
        motor.GetAttribute("newton:kd").Set(0.025)
        motor.GetAttribute("newton:velocityLimit").Set(limit)
    stage.GetRootLayer().Save()
    return path


def _authored_schemas(prim):
    """The API schemas the file authors on `prim`, even one no plugin defines."""
    authored = prim.GetMetadata("apiSchemas")
    return list(authored.GetAddedOrExplicitItems()) if authored else []


def _actuator_schemas(path):
    """The schemas the vehicle at `path` authors on each of its `NewtonActuator` prims."""
    stage = Usd.Stage.Open(str(path))  # held: its prims expire with it
    return {str(p.GetPath()): _authored_schemas(p) for p in stage.Traverse() if p.GetTypeName() == "NewtonActuator"}


def test_the_converter_authors_the_propeller_schema_on_each_rotor_body_and_keeps_the_actuator_prims(tmp_path):
    """The asset converter keeps each `NewtonActuator` prim with Newton's schemas alone and authors the
    propeller schema on each rotor body.

    Given `scripts/assets/convert.py` run on a source vehicle, when it finishes, then each actuator prim
    carries only Newton's schemas, each rotor body applies the propeller schema, and no joint authors
    `propeller:*`. The source authors Newton's schemas alone on its three actuator prims, so they must
    come out as they went in, and its gimbal, a joint Newton drives that declares no rotor, gains none.
    """
    source = _source_vehicle(tmp_path / "source.usda")
    before = _actuator_schemas(source)
    out = convert.author_rotor_params(source, tmp_path / "out.usda", params=PARAMS)

    stage = Usd.Stage.Open(str(out))  # held: its prims expire with it
    prims = list(stage.Traverse())
    bodies = {
        str(p.GetPath()): "NexusPropellerAPI" in _authored_schemas(p)
        for p in prims
        if "PhysicsRigidBodyAPI" in _authored_schemas(p)
    }
    on_joints = [
        f"{p.GetPath()}.{attr.GetName()}"
        for p in prims
        if p.GetTypeName() == "PhysicsRevoluteJoint"
        for attr in p.GetAttributes()
        if attr.GetName().startswith("propeller:")
    ]
    assert (len(before), _actuator_schemas(out), bodies, on_joints) == (
        3,
        before,
        {"/Vehicle/body": False, "/Vehicle/gimbal": False, "/Vehicle/rotor_0": True, "/Vehicle/rotor_1": True},
        [],
    )


def _rotor_bodies(path):
    """The rigid bodies of the vehicle at `path` that apply the propeller schema."""
    stage = Usd.Stage.Open(str(path))  # held: its prims expire with it
    return sorted(str(p.GetPath()) for p in stage.Traverse() if "NexusPropellerAPI" in _authored_schemas(p))


def test_the_converter_takes_the_rotors_a_caller_names(tmp_path):
    """The converter authors the propeller schema on the bodies of the rotor joints a caller names.

    Given a source that declares no rotor, when the converter runs with the two rotor joints named, then
    their two bodies apply the propeller schema and the gimbal's doesn't.
    """
    source = _source_vehicle(tmp_path / "source.usda", declared=False)
    rotors = ["/Vehicle/rotor_0_joint", "/Vehicle/rotor_1_joint"]
    out = convert.author_rotor_params(source, tmp_path / "out.usda", params=PARAMS, rotors=rotors)
    assert _rotor_bodies(out) == ["/Vehicle/rotor_0", "/Vehicle/rotor_1"]
