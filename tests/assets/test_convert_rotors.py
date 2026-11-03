"""``scripts/assets/convert.py`` authors the propeller schema on each rotor's rigid body and leaves each
``NewtonActuator`` prim with Newton's schemas alone, so the converted vehicle declares its rotors the
way the run reads them. A rotor is a joint the source declares as one, or one the caller names: a
joint Newton drives for another purpose, such as a gimbal servo, is none.
"""

import importlib.util
import pathlib

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

ROOT = pathlib.Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location("convert", ROOT / "scripts" / "assets" / "convert.py")
convert = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convert)

PARAMS = {"ct": 3.463e-6, "cd": 0.05, "aero_h": 0.0, "aero_hforce": 0.0}
NEWTON = ["NewtonPIDControlAPI", "NewtonDCMotorClampingAPI"]


def _source_vehicle(path, *, declared=True, limit=397.935):
    """A vehicle in the shape of the pinned ones: two rotor bodies and a gimbal on revolute joints, each joint
    driven by a `NewtonActuator` prim with Newton's schemas. The gimbal is no rotor.

    Args:
        path: Where to write the vehicle.
        declared: Whether the two rotor joints author `propeller:*`, the declaration the schema replaces, and
            `motor:tau`, a motor lag the run no longer reads.
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
            declared_values = {
                "propeller:ct": 3.463e-6,
                "propeller:cd": 0.05,
                "propeller:rpm_max": 3800.0,
                "motor:tau": 0.033,
            }
            for attr, value in declared_values.items():
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
    `propeller:*` or `motor:*`. The source authors Newton's schemas alone on its three actuator prims, so they
    must come out as they went in, and its gimbal, a joint Newton drives that declares no rotor, gains none.
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
        if attr.GetName().startswith(("propeller:", "motor:"))
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


def test_the_converter_fails_when_the_declared_top_speed_and_the_motors_no_load_speed_differ(tmp_path):
    """The converter fails when the speed a source declares and its motor's no-load speed differ.

    Given a source whose rotor joints declare 3800 rpm and whose motors declare 300 rad/s, 2865 rpm, when
    the converter runs, then it fails and names the motor and both speeds, since the run reads the motor's.
    """
    source = _source_vehicle(tmp_path / "source.usda", limit=300.0)
    with pytest.raises(ValueError) as e:
        convert.author_rotor_params(source, tmp_path / "out.usda", params=PARAMS)
    assert all(text in str(e.value) for text in ("/Vehicle/rotor_0_motor", "3800", "300"))


# The ESC's duty map and bus and the motor's constants the converter authors beside the propeller's values, #386.
ROTOR_PARAMS = {
    **PARAMS,
    "idle_duty": 0.1,
    "max_duty": 1.0,
    "bus_voltage": 24.0,
    "kv": 160.0,
    "resistance": 0.14,
    "current_limit": 130.0,
}
ROTOR_ATTRS = {
    "NexusEscAPI": {"nexus:idleDuty": 0.1, "nexus:maxDuty": 1.0, "nexus:busVoltage": 24.0},
    "NexusMotorAPI": {"nexus:kv": 160.0, "nexus:resistance": 0.14, "nexus:currentLimit": 130.0},
    "NexusPropellerAPI": {"nexus:ct": 3.463e-6, "nexus:cd": 0.05, "nexus:aeroH": 0.0, "nexus:aeroHforce": 0.0},
}


def _rotor_declarations(path) -> dict:
    """Each rotor body's authored values under the three rotor schemas, by body, and the joints the vehicle's
    `NewtonActuator` prims target.
    """
    stage = Usd.Stage.Open(str(path))
    prims = list(stage.Traverse())
    bodies = {}
    for prim in prims:
        if str(prim.GetPath()) not in ("/Vehicle/rotor_0", "/Vehicle/rotor_1"):
            continue
        schemas = _authored_schemas(prim)
        bodies[str(prim.GetPath())] = {
            schema: {name: round(prim.GetAttribute(name).Get(), 9) for name in attrs if prim.HasAttribute(name)}
            for schema, attrs in ROTOR_ATTRS.items()
            if schema in schemas
        }
    driven = sorted(
        str(target)
        for prim in prims
        if prim.GetTypeName() == "NewtonActuator"
        for target in prim.GetRelationship("newton:targets").GetTargets()
    )
    return {"bodies": bodies, "driven": driven}


def test_the_converter_authors_each_rotors_three_parts_and_keeps_no_actuator_prim_on_a_rotor_joint(tmp_path):
    """The converter's output declares each rotor's ESC, motor and propeller from the values the caller gives
    and keeps no `NewtonActuator` prim on a rotor joint.

    Given `scripts/assets/convert.py` run on a source vehicle with the ESC's, the motor's and the propeller's
    values, when it finishes, then each rotor body applies the three schemas with those values, and the only
    `NewtonActuator` prim left drives the gimbal.
    """
    source = _source_vehicle(tmp_path / "source.usda")
    out = convert.author_rotor_params(source, tmp_path / "out.usda", params=ROTOR_PARAMS)

    expected = {name: {k: round(v, 9) for k, v in attrs.items()} for name, attrs in ROTOR_ATTRS.items()}
    assert _rotor_declarations(out) == {
        "bodies": {"/Vehicle/rotor_0": expected, "/Vehicle/rotor_1": expected},
        "driven": ["/Vehicle/gimbal_joint"],
    }
