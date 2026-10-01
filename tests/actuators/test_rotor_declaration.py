"""Each rotor declares its propeller on its ``NewtonActuator`` prim, beside Newton's motor, and the run
finds its rotors by that declaration.

Real builds on the Warp CPU backend: a quad authored per test, four rotor bodies on revolute joints
under one airframe, each with a ``NewtonActuator`` prim carrying Newton's velocity servo and DC motor
clamp and the nexus propeller schema, and a stand-in controller that commands full throttle.
Skipped without newton or pxr.
"""

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import warp as wp
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from nexus._src.api.sim import Sim
from nexus._src.build.assembly import build_orchestrator, build_scenario
from nexus._src.core.schema import Controls
from nexus._src.core.stages import peer_stages
from nexus._src.physics.builders.usd import USDBuilder

ROOT = "/Vehicle"
ARM = 0.2  # rotor offset from the airframe's center on each axis [m]
LIFT_CT = 4e-7  # N/rpm²: four rotors at full command make about twice the 1.08 kg vehicle's weight
WEAK_CT = 1e-7  # N/rpm²: four rotors at full command make about half its weight
FLIGHT_STEPS = 125  # 0.5 s at the default 4 ms tick
PROPELLER = {"nexus:cd": 0.05, "nexus:rpmMax": 3800.0, "nexus:aeroH": 0.0, "nexus:aeroHforce": 0.0}


def _author(
    path,
    *,
    ct=LIFT_CT,
    propeller=True,
    odd_ct=None,
    legacy_attr=False,
    stray=False,
    gimbal=False,
    reparent=False,
):
    """A quad whose rotors each declare Newton's motor and the propeller schema on their actuator prim.

    Args:
        path: Where to write the vehicle.
        ct: The thrust coefficient every rotor declares.
        propeller: Whether the actuator prims apply the propeller schema.
        odd_ct: A thrust coefficient rotor 2 declares instead of `ct`.
        legacy_attr: Also author `propeller:ct` on rotor 0's joint.
        stray: Also apply the propeller schema to a prim that targets no joint.
        gimbal: Add a fifth body on a revolute joint that no actuator prim declares.
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
            apply_propeller(motor, odd_ct if odd_ct is not None and i == 2 else ct)
    if gimbal:
        offset = (0.0, 0.0, 0.12)
        mount = child_body("gimbal", offset)
        revolute("gimbal_joint", f"{ROOT}/body", mount.GetPath(), offset)
    if stray:
        apply_propeller(UsdGeom.Xform.Define(stage, f"{ROOT}/stray").GetPrim(), ct)
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
def test_each_rotors_thrust_comes_from_the_propeller_schema_on_its_actuator_prim(tmp_path, ct, climbs):
    """Each rotor's thrust and drag values come from the propeller schema on its `NewtonActuator` prim.

    Given a fixture vehicle whose actuator prims apply the propeller schema with a distinctive `ct`, when
    the run builds it, then the actuator flies with that `ct`: at full throttle, a `ct` that lifts twice
    the weight climbs half a meter in half a second, and one that lifts half of it stays down.
    """
    assert (_climb(_author(tmp_path / "quad.usda", ct=ct)) > 0.5) is climbs


def test_a_revolute_joint_no_actuator_prim_declares_is_not_a_rotor(tmp_path):
    """A revolute joint that no actuator prim declares as a rotor isn't a rotor.

    Given the fixture with a fifth revolute joint and no propeller schema on it, when built, then the run
    has four rotors and the fifth joint carries no thrust: four commands at full throttle fly it up.
    """
    assert _climb(_author(tmp_path / "quad.usda", gimbal=True)) > 0.5


def test_a_vehicle_that_declares_no_rotor_fails_the_build_and_names_the_vehicle(tmp_path):
    """A vehicle that declares no rotor fails the build and names the vehicle.

    Given the fixture with every propeller schema removed, when built, then the build fails naming the
    vehicle's root prim.
    """
    assert ROOT in _build_error(_author(tmp_path / "quad.usda", propeller=False))


def test_a_propeller_schema_on_a_prim_that_targets_no_joint_fails_the_build_and_names_the_prim(tmp_path):
    """A propeller schema on a prim that targets no joint fails the build and names the prim.

    Given the fixture with the propeller schema on a prim without `newton:targets`, when built, then the
    build fails naming that prim.
    """
    assert f"{ROOT}/stray" in _build_error(_author(tmp_path / "quad.usda", stray=True))


def test_rotors_that_do_not_share_one_parent_body_fail_the_build_and_name_the_prims(tmp_path):
    """Rotors that don't share one parent body fail the build and name the prims.

    Given the fixture with one rotor joint re-parented to another body, when built, then the build fails
    naming the rotor prims and their parents.
    """
    message = _build_error(_author(tmp_path / "quad.usda", reparent=True))
    assert all(name in message for name in (f"{ROOT}/rotor_3_motor", f"{ROOT}/rotor_0", f"{ROOT}/body"))


def test_rotors_whose_propeller_values_differ_fail_the_build_and_name_the_prims(tmp_path):
    """Rotors whose propeller values differ fail the build and name the prims.

    Given the fixture with one rotor's `ct` changed, when built, then the build fails naming the rotors
    whose values differ.
    """
    message = _build_error(_author(tmp_path / "quad.usda", odd_ct=2 * LIFT_CT))
    assert all(f"{ROOT}/rotor_{i}_motor" in message for i in (0, 2))


def test_a_vehicle_that_still_authors_propeller_joint_attributes_fails_the_build_and_names_the_prim(tmp_path):
    """A vehicle that still authors `propeller:*` joint attributes fails the build and names the prim.

    Given the fixture with `propeller:ct` authored on a rotor joint, when built, then the build fails naming
    the joint and the attribute.
    """
    message = _build_error(_author(tmp_path / "quad.usda", legacy_attr=True))
    assert f"{ROOT}/rotor_0_joint" in message and "propeller:ct" in message


def test_newton_still_builds_each_rotors_motor_from_the_actuator_prim_that_carries_the_propeller_schema(tmp_path):
    """Newton still builds each rotor's motor from the actuator prim that carries the propeller schema.

    Given the fixture, when built, then `model.actuators` holds one Proportional Integral Derivative (PID) controlled, DC-clamped motor per
    rotor.
    """
    with wp.ScopedDevice("cpu"):
        model = _build(_author(tmp_path / "quad.usda")).physics.model
    motors = [
        (type(a.controller).__name__, tuple(type(c).__name__ for c in a.clamping))
        for a in model.actuators
        for _ in range(a.num_actuators)
    ]
    assert motors == [("ControllerPID", ("ClampingDCMotor",))] * 4
