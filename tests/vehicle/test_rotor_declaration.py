"""Each rotor declares its propeller on its rigid body, beside Newton's motor, and the run finds its
rotors by that declaration.

Real builds on the Warp CPU backend: a quad authored per test, four rotor bodies on revolute joints
under one airframe, each body carrying the nexus propeller schema and each joint driven by a
``NewtonActuator`` prim with Newton's velocity servo and DC motor clamp, and a stand-in controller that
commands full throttle, from ``tests/vehicle/quad.py``. Skipped without newton or pxr.
"""

import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import newton
import warp as wp

from nexus_sim._src.physics.builders.usd import USDBuilder, parse_rotors
from nexus_sim._src.vehicle.rotors import find_rotor_joints
from tests.vehicle.quad import LIFT_CT, ROOT, WEAK_CT
from tests.vehicle.quad import author as _author
from tests.vehicle.quad import build as _build
from tests.vehicle.quad import build_error as _build_error
from tests.vehicle.quad import climb as _climb


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
