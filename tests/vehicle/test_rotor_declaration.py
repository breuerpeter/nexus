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

from nexus_uav._src.physics.vehicle import Rotor, VehicleUsd, rotors
from nexus_uav._src.vehicle.rotors import find_rotor_joints
from tests.vehicle.quad import FLIGHT_STEPS, LIFT_CT, ROOT, WEAK_CT
from tests.vehicle.quad import author as _author
from tests.vehicle.quad import build as _build
from tests.vehicle.quad import build_error as _build_error
from tests.vehicle.quad import climb as _climb
from tests.vehicle.quad import fly as _fly
from tests.vehicle.quad import tilt as _tilt

CHANNELS = (0, 1, 2, 3)  # each rotor's controls channel, in rotor order
TILTED = 5.0  # how far off level a quad with one stronger rotor ends, at least [degrees]
LEVEL = 1.0  # how far off level a quad with equal rotors ends, at most [degrees]


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


def _final_tilt(path) -> tuple[float | None, str | None]:
    """How far off level the airframe ends [degrees] after the flight at full command, and the build's error
    message when it fails instead.
    """
    try:
        with wp.ScopedDevice("cpu"):
            rows = _fly(_build(path, FLIGHT_STEPS), field="orientation")
    except ValueError as exc:
        return None, str(exc)
    return _tilt(rows[-1]), None


def test_a_vehicle_whose_rotors_declare_different_propeller_values_builds_and_each_rotor_flies_its_own(tmp_path):
    """A vehicle whose rotors declare different propeller values builds, and each rotor flies its own.

    Given the fixture quad with rotor 2's `ct` twice the others', when the run flies 125 ticks at one equal
    command on every rotor, then it builds and the airframe ends tilted, where the quad with equal `ct` ends
    level.
    """
    odd = _author(tmp_path / "odd.usda", odd_ct=2 * LIFT_CT, channels=CHANNELS)
    even = _author(tmp_path / "even.usda", channels=CHANNELS)

    (odd_tilt, odd_error), (even_tilt, even_error) = _final_tilt(odd), _final_tilt(even)

    tilted = odd_tilt is not None and odd_tilt > TILTED
    level = even_tilt is not None and even_tilt < LEVEL
    assert (tilted, level) == (True, True), (odd_tilt, odd_error, even_tilt, even_error)


def test_two_rotors_that_declare_one_controls_channel_fail_the_build_naming_both(tmp_path):
    """Two rotors that declare one controls channel fail the build, naming both.

    Given the fixture quad with rotors 1 and 2 declaring the same channel, when the run builds, then it raises
    naming both rotor prims and the channel, and no stage ran.
    """
    message = _build_error(_author(tmp_path / "quad.usda", channels=(0, 1, 1, 3)))
    assert all(name in message for name in (f"{ROOT}/rotor_1", f"{ROOT}/rotor_2", "channel"))


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


def test_the_rotor_reader_returns_each_rotor_with_its_own_values_in_channel_order(tmp_path):
    """The rotor reader returns each rotor with its own values, in channel order.

    Given the fixture with its ESCs on channels 3, 2, 1 and 0 and rotor 2's `ct` twice the others', when read,
    then the rotors come in channel order, rotor 3 first, each with its body, its joint, the airframe, its
    channel, its own propeller values and the motor's no-load speed, 398 rad/s or 3800.62 rpm.
    """
    read = rotors(_author(tmp_path / "quad.usda", odd_ct=2 * LIFT_CT, channels=(3, 2, 1, 0)))
    expected = [
        Rotor(
            body=f"{ROOT}/rotor_{i}",
            joint=f"{ROOT}/rotor_{i}_joint",
            parent=f"{ROOT}/body",
            channel=3 - i,
            propeller={"ct": 2 * LIFT_CT if i == 2 else LIFT_CT, "cd": 0.05, "aero_h": 0.0, "aero_hforce": 0.0},
            rpm_max=3800.62,
        )
        for i in (3, 2, 1, 0)
    ]
    assert [(r.body, r.joint, r.parent, r.channel) for r in read] == [
        (r.body, r.joint, r.parent, r.channel) for r in expected
    ]
    assert [r.propeller for r in read] == [pytest.approx(r.propeller, rel=1e-5) for r in expected]
    assert [r.rpm_max for r in read] == pytest.approx([r.rpm_max for r in expected], rel=1e-5)


def test_a_rotor_that_declares_a_propeller_and_no_esc_fails_the_build_and_names_the_prim(tmp_path):
    """A rotor that declares a propeller and no ESC fails the build and names the prim.

    Given the fixture with no ESC on any rotor, when built, then the build fails naming rotor 0 and the ESC's
    schema.
    """
    message = _build_error(_author(tmp_path / "quad.usda", channels=None))
    assert f"{ROOT}/rotor_0" in message and "NexusEscAPI" in message


def test_find_rotor_joints_returns_the_declared_joints_and_no_other(tmp_path):
    """`find_rotor_joints` returns the model's joints the vehicle declares as rotors, and no other.

    Given the fixture with a fifth revolute joint, built into a model, when asked for the four declared
    joints, then it returns the four rotor bodies, on the airframe.
    """
    with wp.ScopedDevice("cpu"):
        builder = newton.ModelBuilder()
        VehicleUsd({"usd_path": str(_author(tmp_path / "quad.usda", gimbal=True))}).build(builder)
        model = builder.finalize()
    _vel, _pos, bodies, base = find_rotor_joints(model, [f"{ROOT}/rotor_{i}_joint" for i in range(4)])
    labels = list(model.body_label)
    assert ([labels[b] for b in bodies], labels[base]) == ([f"{ROOT}/rotor_{i}" for i in range(4)], f"{ROOT}/body")
