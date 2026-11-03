"""Each rotor declares its ESC, its motor and its propeller on its rigid body, and the run finds its rotors by
that declaration.

Real builds on the Warp CPU backend: a quad authored per test, four rotor bodies on revolute joints under one
airframe, each body carrying the three rotor schemas, and a stand-in controller that commands full throttle, from
``tests/vehicle/quad.py``. Skipped without newton or pxr.
"""

import math

import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import newton
import numpy as np
import warp as wp

from nexus_uav._src.api.sim import Sim
from nexus_uav._src.physics.vehicle import Rotor, VehicleUsd, rotors
from nexus_uav._src.vehicle.rotors import find_rotor_joints
from tests.vehicle import quad
from tests.vehicle.quad import ESC, FLIGHT_STEPS, LIFT_CT, MOTOR, PROPELLER, ROOT, WEAK_CT
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


def test_the_rotor_reader_returns_each_rotor_with_its_own_values_in_channel_order(tmp_path):
    """The rotor reader returns each rotor with its own values, in channel order.

    Given the fixture with its ESCs on channels 3, 2, 1 and 0 and rotor 2's `ct` twice the others', when read,
    then the rotors come in channel order, rotor 3 first, each with its body, its joint, the airframe, its
    channel and its own ESC, motor and propeller values.
    """
    read = rotors(_author(tmp_path / "quad.usda", odd_ct=2 * LIFT_CT, channels=(3, 2, 1, 0)))
    expected = [
        Rotor(
            body=f"{ROOT}/rotor_{i}",
            joint=f"{ROOT}/rotor_{i}_joint",
            parent=f"{ROOT}/body",
            channel=3 - i,
            esc={"idle_duty": 0.1, "max_duty": 1.0, "bus_voltage": 24.0},
            motor={"kv": 160.0, "resistance": 0.28, "current_limit": 130.0},
            propeller={"ct": 2 * LIFT_CT if i == 2 else LIFT_CT, "cd": 0.05, "aero_h": 0.0, "aero_hforce": 0.0},
        )
        for i in (3, 2, 1, 0)
    ]
    assert [(r.body, r.joint, r.parent, r.channel) for r in read] == [
        (r.body, r.joint, r.parent, r.channel) for r in expected
    ]
    assert [(r.esc, r.motor, r.propeller) for r in read] == [
        (pytest.approx(r.esc, rel=1e-5), pytest.approx(r.motor, rel=1e-5), pytest.approx(r.propeller, rel=1e-5))
        for r in expected
    ]


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


# The rotor chain of the fixture quad with `motor="nexus"`, #386: the ESC's duty map times its bus is the
# terminal voltage, and the motor's torque is the first-order DC motor's, `kd·(V/Ke − Ω)` with `kd = Ke²/R`.
RPM_PER_RADS = 60.0 / (2.0 * math.pi)
SETTLED = 0.01  # how far off the loaded speed, the voltage and the current a settled rotor can read, as a fraction


def _ke() -> float:
    """The fixture motor's back electromotive force constant [V·s/rad], from its `kv` [rpm/V]; `Kt` [N·m/A] is the same number."""
    return 60.0 / (2.0 * math.pi * MOTOR["nexus:kv"])


def _voltage(command: float) -> float:
    """The fixture ESC's terminal voltage under `command` [V]: its duty map times its bus."""
    duty = ESC["nexus:idleDuty"] + (ESC["nexus:maxDuty"] - ESC["nexus:idleDuty"]) * command
    return duty * ESC["nexus:busVoltage"]


def _loaded_speed(command: float) -> float:
    """The speed a fixture rotor settles at under `command` [rad/s]: the positive root of
    `kd·(V/Ke − Ω) = cd·kf·Ω²`, the motor's torque against the propeller's drag, with `kf = ct·RPM_PER_RADS²`.
    """
    ke = _ke()
    kd = ke * ke / MOTOR["nexus:resistance"]
    a = PROPELLER["nexus:cd"] * LIFT_CT * RPM_PER_RADS**2
    b, c = kd, -kd * _voltage(command) / ke
    return (-b + math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)


def _settled_rotor(path, command: float) -> tuple[dict | None, str | None]:
    """Rotor 0's recorded speed [rad/s], its ESC's voltage [V] and its motor's current [A] at the end of a flight of
    the vehicle at `path` at `command` on every rotor, and the error message when the build fails instead.
    """
    try:
        with (
            wp.ScopedDevice("cpu"),
            Sim._from_loop(quad.build(path, FLIGHT_STEPS, quad.Commands([command] * 4))) as sim,
        ):
            sim.run()
            return {
                "speed": float(sim["vehicle/joints/rotor_0_joint"].history()[-1]["qd"][0]),
                "voltage": float(sim["vehicle/commands/rotor_0/voltage"].history()[-1]["voltage"]),
                "current": float(sim["vehicle/actuators/rotor_0/current"].history()[-1]["current"]),
            }, None
    except ValueError as exc:
        return None, str(exc)


def _within(value: float | None, expected: float) -> bool:
    return value is not None and abs(value - expected) <= SETTLED * abs(expected)


@pytest.mark.parametrize("command", [0.0, 1.0])
def test_a_rotor_settles_at_the_loaded_speed_its_escs_duty_map_and_its_motors_constants_give(tmp_path, command):
    """A rotor at command 0 settles at its idle speed, the loaded speed the duty map's `idleDuty` gives, and at
    command 1 at the loaded speed where `kd·(V/Ke − Ω)` meets `cd·kf·Ω²`, with no feedforward holding it up.

    Given the fixture quad with its ESCs' duty map, bus and its motors' constants, when a stand-in controller
    holds `command` on every rotor for 125 ticks, then rotor 0's speed is within 1 % of that root.
    """
    rotor, error = _settled_rotor(_author(tmp_path / "quad.usda", motor="nexus"), command)
    speed = None if rotor is None else rotor["speed"]
    assert _within(speed, _loaded_speed(command)), (speed, _loaded_speed(command), error)


def test_each_escs_voltage_and_each_motors_current_are_in_the_recording_under_the_rotors_path(tmp_path):
    """Each ESC's voltage, `d(u)·busVoltage`, and each motor's current, its applied torque over `Kt`, are in the
    recording under the rotor's path in its role's folder.

    Given that quad at full command for 125 ticks, when a test reads the recording, then rotor 0's `voltage` is its
    bus, and its `current` is the propeller's drag torque at the loaded speed over `Kt`, each within 1 %.
    """
    rotor, error = _settled_rotor(_author(tmp_path / "quad.usda", motor="nexus"), 1.0)
    speed = _loaded_speed(1.0)
    current = PROPELLER["nexus:cd"] * LIFT_CT * RPM_PER_RADS**2 * speed * speed / _ke()
    read = (None, None) if rotor is None else (rotor["voltage"], rotor["current"])
    assert (_within(read[0], _voltage(1.0)), _within(read[1], current)) == (True, True), (read, error)


def test_a_rotor_body_that_applies_two_of_the_three_rotor_schemas_fails_the_build_and_names_the_missing_one(tmp_path):
    """A rotor body that applies two of the ESC's, the motor's and the propeller's schemas without the third fails
    the build, naming the prim and the missing schema.

    Given the fixture quad with the motor's schema left off rotor 1, when built, then the build fails naming
    rotor 1 and `NexusMotorAPI`.
    """
    message = _build_error(_author(tmp_path / "quad.usda", motor="nexus", no_motor_schema=1))
    assert f"{ROOT}/rotor_1" in message and "NexusMotorAPI" in message, message


@pytest.mark.parametrize("attribute", [*sorted(MOTOR), "nexus:busVoltage"])
def test_a_motor_constant_or_a_bus_voltage_that_is_not_positive_fails_the_build_and_names_the_attribute(
    tmp_path, attribute
):
    """A motor whose `nexus:kv`, `nexus:resistance` or `nexus:currentLimit`, or an ESC whose `nexus:busVoltage`, is
    not positive fails the build, naming the prim and the attribute.

    Given the fixture quad with rotor 1's `attribute` authored 0, when built, then the build fails naming rotor 1
    and the attribute.
    """
    message = _build_error(_author(tmp_path / "quad.usda", motor="nexus", zero=(1, attribute)))
    assert f"{ROOT}/rotor_1" in message and attribute in message, message


def test_a_newton_actuator_prim_on_a_rotors_joint_fails_the_build_and_names_the_prim(tmp_path):
    """A `NewtonActuator` prim on a rotor's joint fails the build, naming the prim, since the motor is nexus's.

    Given the fixture quad with its motors declared on the rotor bodies and a `NewtonActuator` prim kept on
    rotor 2's joint, when built, then the build fails naming that prim.
    """
    message = _build_error(_author(tmp_path / "quad.usda", motor="nexus", newton_motor=2))
    assert f"{ROOT}/rotor_2_motor" in message, message


def test_a_rotor_whose_body_lists_the_motors_schema_first_flies_the_same_trajectory(tmp_path):
    """A rotor whose body lists the motor's schema before the ESC's flies the same trajectory as one that lists
    the ESC's first, since the motor runs after the command elements as an actuator.

    Given the fixture quad authored both ways, when each flies 125 ticks at full command, then the recorded
    positions are equal.
    """
    usual = _author(tmp_path / "usual.usda")
    reversed_order = _author(tmp_path / "reversed.usda", motor_first=True)
    with wp.ScopedDevice("cpu"):
        flights = [_fly(_build(path, FLIGHT_STEPS)) for path in (usual, reversed_order)]
    assert np.array_equal(flights[0], flights[1])


@pytest.mark.parametrize(
    ("bad", "attribute"),
    [
        ({"nexus:idleDuty": -0.1}, "nexus:idleDuty"),
        ({"nexus:idleDuty": 1.0, "nexus:maxDuty": 1.0}, "nexus:idleDuty"),
        ({"nexus:maxDuty": 1.5}, "nexus:maxDuty"),
    ],
    ids=["idle below 0", "idle at max", "max above 1"],
)
def test_an_esc_whose_duty_map_is_outside_its_range_fails_the_build_and_names_the_attribute(tmp_path, bad, attribute):
    """An ESC whose duty map is outside `0 ≤ idleDuty < maxDuty ≤ 1` fails the build, naming the prim and the
    attribute.

    Given the fixture quad with every ESC's `bad` values, when built, then the build fails naming rotor 0 and the
    attribute.
    """
    message = _build_error(_author(tmp_path / "quad.usda", esc=bad))
    assert f"{ROOT}/rotor_0" in message and attribute in message, message


def test_an_esc_whose_channel_is_below_0_fails_the_build_and_names_the_attribute(tmp_path):
    """An ESC whose channel is below 0 fails the build, naming the prim and the attribute.

    Given the fixture quad with rotor 1's ESC on channel -1, when built, then the build fails naming rotor 1 and
    `nexus:channel`.
    """
    message = _build_error(_author(tmp_path / "quad.usda", channels=(0, -1, 2, 3)))
    assert f"{ROOT}/rotor_1" in message and "nexus:channel" in message, message
