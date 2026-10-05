"""Each rotor declares its propeller on its rigid body, beside Newton's motor, and the run finds its
rotors by that declaration.

Real builds on the Warp CPU backend: a quad authored per test, four rotor bodies on revolute joints
under one airframe, each body carrying the nexus propeller schema and each joint driven by a
``NewtonActuator`` prim with Newton's velocity servo and DC motor clamp, and a stand-in controller that
commands full throttle. The motor's no-load speed, 398 rad/s, is the rotor speed at full command.
Skipped without newton or pxr.
"""

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import newton
import warp as wp
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from nexus._src.api.sim import Sim
from nexus._src.build.assembly import build_orchestrator, build_scenario
from nexus._src.core.schema import Controls
from nexus._src.core.stages import peer_stages
from nexus._src.physics.builders.usd import USDBuilder, parse_rotors
from nexus._src.vehicle.actuators import find_rotor_joints

ROOT = "/Vehicle"
ARM = 0.2  # rotor offset from the airframe's center on each axis [m]
LIFT_CT = 4e-7  # N/rpm²: four rotors at full command make about twice the 1.08 kg vehicle's weight
WEAK_CT = 1e-7  # N/rpm²: four rotors at full command make about half its weight
FLIGHT_STEPS = 125  # 0.5 s at the default 4 ms tick
PROPELLER = {"nexus:cd": 0.05, "nexus:aeroH": 0.0, "nexus:aeroHforce": 0.0}


def _author(
    path,
    *,
    ct=LIFT_CT,
    propeller=True,
    odd_ct=None,
    legacy_attr=False,
    pod=False,
    gimbal=False,
    reparent=False,
):
    """A quad whose rotor bodies each declare the propeller schema, with Newton's motor on each rotor joint.

    Args:
        path: Where to write the vehicle.
        ct: The thrust coefficient every rotor declares.
        propeller: Whether the rotor bodies apply the propeller schema.
        odd_ct: A thrust coefficient rotor 2 declares instead of `ct`.
        legacy_attr: Also author `propeller:ct` on rotor 0's joint.
        pod: Add a fifth body on a fixed joint that applies the propeller schema.
        gimbal: Add a fifth body on a revolute joint that applies no propeller schema.
        reparent: Hang rotor 3's joint off rotor 0's body instead of the airframe.
    """
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, ROOT).GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    body = UsdGeom.Cube.Define(stage, f"{ROOT}/body")
    body.GetSizeAttr().Set(0.2)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).GetMassAttr().Set(1.0)
    imu = UsdGeom.Xform.Define(stage, f"{ROOT}/body/Imu").GetPrim()
    imu.CreateAttribute("sensor:type", Sdf.ValueTypeNames.Token, custom=True).Set("imu")

    def child_body(name, offset, flipped=False):
        prim = UsdGeom.Cylinder.Define(stage, f"{ROOT}/{name}")
        prim.GetRadiusAttr().Set(0.1)
        prim.GetHeightAttr().Set(0.01)
        prim.AddTranslateOp().Set(Gf.Vec3d(*offset))
        if flipped:  # the spin axis points the other way: the rotor turns the other way for the same command
            prim.AddRotateXOp().Set(180.0)
        UsdPhysics.RigidBodyAPI.Apply(prim.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(prim.GetPrim())
        mass.GetMassAttr().Set(0.02)
        mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(5e-5, 5e-5, 1e-4))
        return prim

    def revolute(name, parent, child, offset, flipped=False):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{ROOT}/{name}")
        joint.CreateBody0Rel().SetTargets([parent])
        joint.CreateBody1Rel().SetTargets([child])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*offset))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        if flipped:
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(0.0, 1.0, 0.0, 0.0))
        joint.CreateAxisAttr().Set("Z")
        return joint.GetPrim()

    def apply_propeller(prim, rotor_ct):
        prim.AddAppliedSchema("NexusPropellerAPI")
        prim.CreateAttribute("nexus:ct", Sdf.ValueTypeNames.Float).Set(rotor_ct)
        for name, value in PROPELLER.items():
            prim.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(value)

    for i, (x, y) in enumerate([(ARM, ARM), (-ARM, -ARM), (ARM, -ARM), (-ARM, ARM)]):
        offset = (x, y, -0.12)  # rotors on the airframe's -z, up once the run spawns it upright
        flipped = bool(i % 2)
        rotor = child_body(f"rotor_{i}", offset, flipped)
        parent = f"{ROOT}/rotor_0" if reparent and i == 3 else f"{ROOT}/body"
        joint = revolute(f"rotor_{i}_joint", parent, rotor.GetPath(), offset, flipped)
        if legacy_attr and i == 0:
            joint.CreateAttribute("propeller:ct", Sdf.ValueTypeNames.Float, custom=True).Set(ct)
        motor = stage.DefinePrim(f"{ROOT}/rotor_{i}_motor", "NewtonActuator")
        motor.CreateRelationship("newton:targets").SetTargets([joint.GetPath()])
        motor.ApplyAPI("NewtonPIDControlAPI")
        motor.ApplyAPI("NewtonDCMotorClampingAPI")
        motor.GetAttribute("newton:kd").Set(0.025)
        motor.GetAttribute("newton:saturationEffort").Set(8.0)
        motor.GetAttribute("newton:maxMotorEffort").Set(8.0)
        motor.GetAttribute("newton:velocityLimit").Set(398.0)
        if propeller:
            apply_propeller(rotor.GetPrim(), odd_ct if odd_ct is not None and i == 2 else ct)
    if gimbal:
        offset = (0.0, 0.0, 0.12)
        mount = child_body("gimbal", offset)
        revolute("gimbal_joint", f"{ROOT}/body", mount.GetPath(), offset)
    if pod:
        offset = (0.0, 0.0, -0.12)
        fixed = UsdPhysics.FixedJoint.Define(stage, f"{ROOT}/pod_joint")
        fixed.CreateBody0Rel().SetTargets([f"{ROOT}/body"])
        fixed.CreateBody1Rel().SetTargets([child_body("pod", offset).GetPath()])
        fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(*offset))
        fixed.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        apply_propeller(stage.GetPrimAtPath(f"{ROOT}/pod"), ct)
    stage.GetRootLayer().Save()
    return path


class _FullThrottle:
    """Answers the preroll at once and commands every rotor to full throttle each tick."""

    def connect(self):
        pass

    def stages(self):
        return peer_stages(self)

    def exchange(self, meas, t, timeout=None):
        return Controls(command=np.ones(4))

    def close(self):
        pass


def _build(path, steps=0):
    """The run the vehicle at `path` builds, around the full-throttle controller."""
    cfg = build_scenario()
    cfg["physics"]["force_cpu"] = True
    return build_orchestrator(
        "quad", cfg, USDBuilder({"usd_path": str(path)}, None), controller=_FullThrottle(), max_steps=steps
    )


def _climb(path):
    """How far the airframe rises over the flight at full throttle [m]."""
    with wp.ScopedDevice("cpu"), Sim.from_orchestrator(_build(path, FLIGHT_STEPS)) as sim:
        sim.run()
        rows = sim.physics["body"].history()
    return rows[-1].position[2] - rows[0].position[2]


def _build_error(path):
    """The message of the error the build of the vehicle at `path` raises."""
    with wp.ScopedDevice("cpu"), pytest.raises(ValueError) as e:
        _build(path)
    return str(e.value)


@pytest.mark.parametrize(("ct", "climbs"), [(LIFT_CT, True), (WEAK_CT, False)])
def test_each_rotors_thrust_comes_from_the_propeller_schema_on_its_rigid_body(tmp_path, ct, climbs):
    """Each rotor's thrust and drag values come from the propeller schema on its rigid body.

    Given a fixture vehicle whose rotor bodies apply the propeller schema with a distinctive `ct`, when
    the run builds it, then the actuator flies with that `ct`: at full throttle, a `ct` that lifts twice
    the weight climbs half a meter in half a second, and one that lifts half of it stays down.
    """
    assert (_climb(_author(tmp_path / "quad.usda", ct=ct)) > 0.5) is climbs


def test_a_revolute_joint_whose_child_body_declares_no_propeller_is_not_a_rotor(tmp_path):
    """A revolute joint whose child body declares no propeller isn't a rotor.

    Given the fixture with a fifth revolute joint whose body applies no propeller schema, when built, then
    the run has four rotors and the fifth joint carries no thrust: four commands at full throttle fly it up.
    """
    assert _climb(_author(tmp_path / "quad.usda", gimbal=True)) > 0.5


def test_a_vehicle_that_declares_no_rotor_fails_the_build_and_names_the_vehicle(tmp_path):
    """A vehicle that declares no rotor fails the build and names the vehicle.

    Given the fixture with every propeller schema removed, when built, then the build fails naming the
    vehicle's root prim.
    """
    assert ROOT in _build_error(_author(tmp_path / "quad.usda", propeller=False))


def test_a_propeller_schema_off_a_revolute_joints_child_body_fails_the_build_and_names_the_prim(tmp_path):
    """A propeller schema on a prim that isn't the child body of a revolute joint fails the build and names the prim.

    Given the fixture with the propeller schema on a body that no revolute joint connects to a parent, when
    built, then the build fails naming that prim.
    """
    assert f"{ROOT}/pod" in _build_error(_author(tmp_path / "quad.usda", pod=True))


def test_rotors_that_do_not_share_one_parent_body_fail_the_build_and_name_the_prims(tmp_path):
    """Rotors that don't share one parent body fail the build and name the prims.

    Given the fixture with one rotor joint re-parented to another body, when built, then the build fails
    naming the rotor bodies and their parents.
    """
    message = _build_error(_author(tmp_path / "quad.usda", reparent=True))
    assert all(name in message for name in (f"{ROOT}/rotor_3", f"{ROOT}/rotor_0", f"{ROOT}/body"))


def test_rotors_whose_propeller_values_differ_fail_the_build_and_name_the_prims(tmp_path):
    """Rotors whose propeller values differ fail the build and name the prims.

    Given the fixture with one rotor's `ct` changed, when built, then the build fails naming the rotor
    bodies whose values differ.
    """
    message = _build_error(_author(tmp_path / "quad.usda", odd_ct=2 * LIFT_CT))
    assert all(f"{ROOT}/rotor_{i}" in message for i in (0, 2))


def test_a_vehicle_that_still_authors_propeller_joint_attributes_fails_the_build_and_names_the_prim(tmp_path):
    """A vehicle that still authors `propeller:*` joint attributes fails the build and names the prim.

    Given the fixture with `propeller:ct` authored on a rotor joint, when built, then the build fails naming
    the joint and the attribute.
    """
    message = _build_error(_author(tmp_path / "quad.usda", legacy_attr=True))
    assert f"{ROOT}/rotor_0_joint" in message and "propeller:ct" in message


def test_newton_still_builds_each_rotors_motor_from_its_actuator_prim(tmp_path):
    """Newton still builds each rotor's motor from its `NewtonActuator` prim.

    Given the fixture, when built, then `model.actuators` holds, per rotor, one
    Proportional Integral Derivative (PID) controlled, DC-clamped motor.
    """
    with wp.ScopedDevice("cpu"):
        model = _build(_author(tmp_path / "quad.usda")).physics.model
    motors = [
        (type(a.controller).__name__, tuple(type(c).__name__ for c in a.clamping))
        for a in model.actuators
        for _ in range(a.num_actuators)
    ]
    assert motors == [("ControllerPID", ("ClampingDCMotor",))] * 4


def test_the_rotor_reader_returns_the_values_the_rotors_share_and_each_rotors_joint(tmp_path):
    """The rotor reader returns the values the rotors share and each rotor's joint.

    Given the fixture, when read, then the map holds the propeller's values, with the motor's no-load
    speed, 398 rad/s or 3800.62 rpm, as the speed at full command, and the joints are the four rotor joints.
    """
    values = {"ct": LIFT_CT, "cd": 0.05, "aero_h": 0.0, "aero_hforce": 0.0, "rpm_max": 3800.62}
    expected = (pytest.approx(values, rel=1e-5), [f"{ROOT}/rotor_{i}_joint" for i in range(4)])
    assert parse_rotors(_author(tmp_path / "quad.usda")) == expected


def test_find_rotor_joints_returns_the_declared_joints_and_no_other(tmp_path):
    """`find_rotor_joints` returns the model's joints the vehicle declares as rotors, and no other.

    Given the fixture with a fifth revolute joint, built into a model, when asked for the four declared
    joints, then it returns the four rotor bodies, on the airframe.
    """
    with wp.ScopedDevice("cpu"):
        builder = newton.ModelBuilder()
        USDBuilder({"usd_path": str(_author(tmp_path / "quad.usda", gimbal=True))}, None).build(builder)
        model = builder.finalize()
    _vel, _pos, bodies, base = find_rotor_joints(model, [f"{ROOT}/rotor_{i}_joint" for i in range(4)])
    labels = list(model.body_label)
    assert ([labels[b] for b in bodies], labels[base]) == ([f"{ROOT}/rotor_{i}" for i in range(4)], f"{ROOT}/body")
