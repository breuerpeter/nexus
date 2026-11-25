"""The gimbal's law turns the camera's attitude error into one current command per joint motor, for the gimbals
of many vehicles in one step.

Each test builds its chains as Newton models with a fixed base, so the base's frame is the vehicle's, and steps the
law on the Warp CPU backend over the inputs a gimbal's controller measures: the camera's rate in its own frame, its
attitude and the three joint angles. A square chain has a pan joint about z, a roll joint about x and a tilt joint
about y, in that order from the vehicle, so at zero joint angles each joint's axis is one camera axis.
"""

import math

import newton
import numpy as np
import pytest
import warp as wp

from nexus_uav._src.vehicle.controllers.gimbal import GimbalController, GimbalLaw

pytestmark = pytest.mark.usefixtures("warp_cpu")

X, Y, Z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
SQUARE = (Z, X, Y)  # a pan, a roll and a tilt joint, from the vehicle to the camera
PAN, ROLL, TILT = 0, 1, 2  # each joint's place in the square chain, and its entry of the controls
# Each axis's lock bit, the joint of the square chain that turns about it, and its camera axis.
AXES = {"roll": (0, ROLL, X), "pitch": (1, TILT, Y), "yaw": (2, PAN, Z)}
ATTITUDE_GAIN = 5.0  # 1/s
RATE_GAIN = (0.5, 0.4, 0.2)  # the pan, roll and tilt motor's, s/rad
DT = 0.004  # s
LEVEL = (0.0, 0.0, 0.0, 1.0)  # a level camera that faces North
ALL = (True, True, True)  # every axis locked
ATOL = 1e-5  # the kernels compute in float32


def _quat(axis, angle: float) -> tuple:
    """The quaternion `(x, y, z, w)` of a turn of `angle` radians about `axis`."""
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    return (*(axis * math.sin(angle / 2.0)), math.cos(angle / 2.0))


def _matrix(q) -> np.ndarray:
    """The rotation matrix of the quaternion `q`, `(x, y, z, w)`."""
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _model(*chains):
    """A model of one chain of three revolute joints per vehicle, each on a fixed base, and each chain's joints.

    A chain is its joints' axes from the vehicle to the camera and a skew: with a skew other than 0, each joint's
    two frames turn off its bodies' frames, a different turn per joint.
    """
    builder = newton.ModelBuilder()
    joints = []
    for axes, skew in chains:
        chain, parent = [], -1
        for k, axis in enumerate(axes):
            link = builder.add_link(mass=0.1, inertia=wp.mat33(np.eye(3) * 1e-4))
            on_parent = wp.transform((0.0, 0.0, 0.05), wp.quat(*_quat((1.0, 2.0, 3.0), 0.3 * skew * (k + 1))))
            on_child = wp.transform((0.0, 0.0, 0.0), wp.quat(*_quat((3.0, 1.0, 2.0), 0.2 * skew * (k + 1))))
            chain.append(
                builder.add_joint_revolute(parent, link, parent_xform=on_parent, child_xform=on_child, axis=axis)
            )
            parent = link
        builder.add_articulation(chain)
        joints.append(chain)
    return builder.finalize(), joints


def _law(vehicles: int = 1, *, rate_limit: float = 10.0, integral_gain: float = 0.0) -> GimbalLaw:
    """The law over `vehicles` square chains, with the tests' gains and one integral gain for every motor."""
    model, joints = _model(*[(SQUARE, 0.0)] * vehicles)
    return GimbalLaw(
        model,
        wp.array(joints, dtype=wp.int32),
        attitude_gain=ATTITUDE_GAIN,
        rate_limit=rate_limit,
        rate_gain=RATE_GAIN,
        integral_gain=(integral_gain,) * 3,
    )


def _step(law: GimbalLaw, vehicles: int = 1, *, rate=None, attitude=None, angles=None, steps: int = 1) -> np.ndarray:
    """Step `law` `steps` times over the same inputs, one row per vehicle: the camera's `rate` in its own frame, no
    rate by default, its `attitude`, level and facing North by default, and the joint `angles`, 0 by default.
    Returns each step's commands, `(steps, vehicles, 3)`.
    """
    rates = wp.array(np.zeros((vehicles, 3)) if rate is None else rate, dtype=wp.vec3)
    attitudes = wp.array([LEVEL] * vehicles if attitude is None else attitude, dtype=wp.quat)
    joint_angles = wp.array(np.zeros((vehicles, 3)) if angles is None else angles, dtype=float)
    controls = wp.zeros((vehicles, 3), dtype=float)
    history = []
    for _ in range(steps):
        law.step(rates, attitudes, joint_angles, DT, controls)
        history.append(controls.numpy().copy())
    return np.array(history)


def _turned(joint: int, angle: float) -> list[float]:
    """The square chain's joint angles under a level camera that faces North, over a vehicle turned `angle`
    radians about the axis of `joint`: that joint takes up the turn.
    """
    angles = [0.0, 0.0, 0.0]
    angles[joint] = -angle
    return angles


@pytest.mark.parametrize("axis", ["roll", "pitch", "yaw"])
def test_the_attitude_error_taken_the_short_way_round_commands_each_motor_in_proportion(axis):
    """The attitude error, taken the short way round, commands each motor in proportion, for many vehicles in one
    step.

    Given two vehicles with square axes at zero joint angles, every axis locked, no rate, no integral gain, a
    target 0.1 rad about one camera axis on the first and 350° about it on the second, when the law steps once,
    then the motor on that axis gets the rate gain times the attitude gain times `2·sin(0.05)` on the first, and
    on the second the turn of −10°, `2·sin(−5°)`, and the others get 0. Once per camera axis.
    """
    _, joint, about = AXES[axis]
    law = _law(2)
    law.write([_quat(about, 0.1), _quat(about, math.radians(350.0))], [(0.0, 0.0, 0.0)] * 2, [ALL] * 2)

    controls = _step(law, 2)[-1]

    expected = np.zeros((2, 3))
    expected[0, joint] = RATE_GAIN[joint] * ATTITUDE_GAIN * 2.0 * math.sin(0.05)
    expected[1, joint] = RATE_GAIN[joint] * ATTITUDE_GAIN * 2.0 * math.sin(math.radians(-5.0))
    np.testing.assert_allclose(controls, expected, atol=ATOL)


def test_a_rate_command_above_the_limit_is_cut_to_the_limit_as_a_vector():
    """The law cuts a rate command over the limit to the limit as a vector, and keeps its direction.

    Given a rate limit of 2 rad/s, no setpoint rate, and a target 1 rad about the axis (0.6, 0.8, 0) of a level
    camera, so the error lies on the roll and pitch axes in the ratio 3 to 4 and its rate command is 4.8 rad/s,
    when the law steps once, then the roll and tilt motors' commands are the rate gain times 0.6 and 0.8 of the
    limit, and the pan motor's is 0.
    """
    law = _law(rate_limit=2.0)
    law.write([_quat((0.6, 0.8, 0.0), 1.0)], [(0.0, 0.0, 0.0)], [ALL])

    controls = _step(law)[-1]

    expected = [[0.0, RATE_GAIN[ROLL] * 0.6 * 2.0, RATE_GAIN[TILT] * 0.8 * 2.0]]
    np.testing.assert_allclose(controls, expected, atol=ATOL)


def test_a_camera_rate_with_no_attitude_error_commands_each_motor_against_it():
    """A camera rate with no attitude error commands each motor against it.

    Given the camera at its target with a rate of 0.5 rad/s about its x axis, when the law steps once, then the
    roll motor gets minus the rate gain times 0.5, and the others get 0.
    """
    law = _law()

    controls = _step(law, rate=[(0.5, 0.0, 0.0)])[-1]

    np.testing.assert_allclose(controls, [[0.0, -RATE_GAIN[ROLL] * 0.5, 0.0]], atol=ATOL)


def test_the_rate_error_reaches_each_motor_as_its_joints_axis_dotted_with_the_error():
    """The rate error reaches each motor as its joint's axis dotted with the error, for any chain of three revolute
    joints at any joint angles.

    Given two chains built as Newton models, a pan, roll, tilt one with square frames and a tilt, roll, pan one
    with skewed joint frames, at joint angles away from zero with the roll joint at 40°, the camera at its target,
    and a rate error of (0.3, −0.2, 0.5) rad/s, when the law steps once, then each chain's three commands equal
    the rate gain times the transpose of Newton's `eval_jacobian` for the camera's rate, in the camera's frame,
    times the error, within 1e-4.
    """
    model, joints = _model((SQUARE, 0.0), ((Y, X, Z), 1.0))
    angles = np.array([[0.3, math.radians(40.0), -0.5], [-0.4, math.radians(40.0), 0.6]], dtype=np.float32)
    error = np.array([0.3, -0.2, 0.5])
    law = GimbalLaw(
        model,
        wp.array(joints, dtype=wp.int32),
        attitude_gain=ATTITUDE_GAIN,
        rate_limit=10.0,
        rate_gain=RATE_GAIN,
        integral_gain=(0.0, 0.0, 0.0),
    )
    law.write([LEVEL] * 2, [(0.0, 0.0, 0.0)] * 2, [ALL] * 2)

    controls = _step(law, 2, rate=[tuple(-error)] * 2, angles=angles)[-1]

    # The oracle: the angular rows of the camera's link, the third of its chain, turned into the camera's frame.
    state = model.state()
    newton.eval_fk(model, wp.array(angles.reshape(-1), dtype=float), wp.zeros(6, dtype=float), state)
    jacobian = newton.eval_jacobian(model, state).numpy()
    poses = state.body_q.numpy()
    expected = np.zeros((2, 3))
    for vehicle in range(2):
        to_camera = _matrix(poses[3 * vehicle + 2][3:]).T
        axes = to_camera @ jacobian[vehicle][2 * 6 + 3 : 2 * 6 + 6, :]
        expected[vehicle] = np.asarray(RATE_GAIN) * (axes.T @ error)
    np.testing.assert_allclose(controls, expected, atol=1e-4)


def test_a_rate_error_that_stays_grows_each_motors_command_by_the_integral():
    """A rate error that stays grows each motor's command by the integral gain times the error times the tick.

    Given a rate error of 0.1 rad/s on the roll axis, the camera at its target turning at −0.1 rad/s, an integral
    gain of 2 and a 4 ms tick, when the law steps 10 times, then the roll motor's command is the rate gain times
    0.1 plus 0.008, and the others are 0.
    """
    law = _law(integral_gain=2.0)

    controls = _step(law, rate=[(-0.1, 0.0, 0.0)], steps=10)[-1]

    np.testing.assert_allclose(controls, [[0.0, RATE_GAIN[ROLL] * 0.1 + 0.008, 0.0]], atol=ATOL)


def test_a_motors_command_stays_within_one_and_so_does_its_integral():
    """A motor's command stays within −1 to 1, and so does its integral.

    Given an integral gain of 2 and a rate error of 5 rad/s on the roll axis, which holds the roll motor's command
    at the clamp for 500 steps, then one step with no error, then one with an error of −1 rad/s, when the law
    steps, then the command is 1 on each of the 500 steps, 1 on the step with no error, the integral alone, and
    0.592 on the last: the rate gain times −1 plus an integral of 1 less 0.008. An integral past 1 would hold the
    last command at the clamp.
    """
    law = _law(integral_gain=2.0)

    held = _step(law, rate=[(-5.0, 0.0, 0.0)], steps=500)[:, 0, ROLL]
    free = _step(law)[-1, 0, ROLL]
    back = _step(law, rate=[(1.0, 0.0, 0.0)])[-1, 0, ROLL]

    assert (bool(np.all(held == 1.0)), float(free), float(back)) == (True, 1.0, pytest.approx(0.592, abs=ATOL))


@pytest.mark.parametrize("axis", ["roll", "pitch", "yaw"])
def test_a_locked_axis_holds_against_the_horizon_or_north_and_an_unlocked_one_against_the_vehicle(axis):
    """A locked axis holds its target against the horizon or North, and an unlocked axis holds it against the
    vehicle's angle on that axis.

    Given a level camera that faces North with no rate and a target of 0, over a vehicle turned 0.2 rad about one
    axis, on two vehicles, the first with that axis unlocked and the second with it locked, when the law steps
    once, then on the first that axis's motor gets the rate gain times the attitude gain times `2·sin(0.1)`,
    toward the vehicle's angle, and the others get 0, and on the second every motor gets 0. Once per axis, since
    each axis reads a different angle of the vehicle.
    """
    bit, joint, _ = AXES[axis]
    unlocked = tuple(i != bit for i in range(3))
    law = _law(2)
    law.write([LEVEL] * 2, [(0.0, 0.0, 0.0)] * 2, [unlocked, ALL])

    controls = _step(law, 2, angles=[_turned(joint, 0.2)] * 2)[-1]

    expected = np.zeros((2, 3))
    expected[0, joint] = RATE_GAIN[joint] * ATTITUDE_GAIN * 2.0 * math.sin(0.1)
    np.testing.assert_allclose(controls, expected, atol=ATOL)


def test_the_setpoints_rate_moves_the_target_each_tick_and_adds_to_the_rate_command():
    """The setpoint's rate moves the target each tick and adds to the rate command.

    Given a setpoint of 0.5 rad/s about pitch, a locked axis, with a NaN attitude, no measured rate, a 4 ms tick
    and the camera held at its start, when the law steps 25 times, then the tilt motor gets the rate gain times
    the sum of 0.5 and the attitude gain times `2·sin(0.025)`, the target 0.05 rad on, and the others get 0.
    """
    law = _law()
    law.write([(math.nan,) * 4], [(0.0, 0.5, 0.0)], [ALL])

    controls = _step(law, steps=25)[-1]

    tilt = RATE_GAIN[TILT] * (0.5 + ATTITUDE_GAIN * 2.0 * math.sin(0.025))
    np.testing.assert_allclose(controls, [[0.0, 0.0, tilt]], atol=ATOL)


def test_a_written_attitude_replaces_the_target_and_a_nan_attitude_leaves_it():
    """A write of an attitude replaces the target, and a write of a NaN attitude leaves the target where the rate
    has taken it.

    Given the pitch target 0.05 rad on after 25 steps at 0.5 rad/s, the camera held at its start, when a write
    sets the rate to 0 with a NaN attitude and the law steps once, then the tilt motor gets the rate gain times
    the attitude gain times `2·sin(0.025)`; and when a write then sets a level attitude with 0.5 rad/s and the law
    steps 25 times, then the target is 0.05 rad from level again: the tilt motor gets the rate gain times the sum
    of 0.5 and the attitude gain times `2·sin(0.025)`.
    """
    law = _law()
    law.write([(math.nan,) * 4], [(0.0, 0.5, 0.0)], [ALL])
    _step(law, steps=25)

    law.write([(math.nan,) * 4], [(0.0, 0.0, 0.0)], [ALL])
    kept = _step(law)[-1, 0, TILT]
    law.write([LEVEL], [(0.0, 0.5, 0.0)], [ALL])
    replaced = _step(law, steps=25)[-1, 0, TILT]

    error = ATTITUDE_GAIN * 2.0 * math.sin(0.025)
    expected = (RATE_GAIN[TILT] * error, RATE_GAIN[TILT] * (0.5 + error))
    np.testing.assert_allclose((kept, replaced), expected, atol=ATOL)


@pytest.mark.parametrize("axis", ["roll", "pitch", "yaw"])
def test_a_lock_bit_that_changes_with_a_nan_attitude_leaves_the_camera_where_it_points(axis):
    """A lock bit that changes with a NaN attitude leaves the camera where it points, and the target then rides
    the new frame.

    Given two cameras at their targets with no rate, each over a vehicle turned 0.2 rad about one axis, the first
    with that axis locked and level, the second with it unlocked and turned with its vehicle, when a write flips
    the axis on both with a NaN attitude and the law steps once, then every motor gets 0; and when each vehicle
    then turns 0.1 rad more, then on the first, now unlocked, that axis's motor gets the rate gain times the
    attitude gain times `2·sin(0.05)`, toward the vehicle, and on the second, now locked, every motor still gets
    0. Once per axis, since each axis reads a different angle of the vehicle.
    """
    bit, joint, about = AXES[axis]
    unlocked = tuple(i != bit for i in range(3))
    still, nan = [(0.0, 0.0, 0.0)] * 2, [(math.nan,) * 4] * 2
    # The second camera follows its vehicle's 0.2 rad, so its joints sit at 0.
    attitude = [LEVEL, _quat(about, 0.2)]
    law = _law(2)
    law.write([LEVEL] * 2, still, [ALL, unlocked])
    _step(law, 2, attitude=attitude, angles=[_turned(joint, 0.2), _turned(joint, 0.0)])

    law.write(nan, still, [unlocked, ALL])
    at_change = _step(law, 2, attitude=attitude, angles=[_turned(joint, 0.2), _turned(joint, 0.0)])[-1]
    after_turn = _step(law, 2, attitude=attitude, angles=[_turned(joint, 0.3), _turned(joint, 0.1)])[-1]

    expected = np.zeros((2, 2, 3))
    expected[1, 0, joint] = RATE_GAIN[joint] * ATTITUDE_GAIN * 2.0 * math.sin(0.05)
    np.testing.assert_allclose([at_change, after_turn], expected, atol=ATOL)


PRIM = "/vehicle/GimbalController"
CAMERA = "/vehicle/camera"
JOINTS = ["/vehicle/pan_joint", "/vehicle/roll_joint", "/vehicle/tilt_joint"]


def _vehicle():
    """A model of one square chain whose joints and camera carry the paths of a vehicle's prims."""
    builder = newton.ModelBuilder()
    chain, parent = [], -1
    for k, (axis, joint) in enumerate(zip(SQUARE, JOINTS, strict=True)):
        link = builder.add_link(
            mass=0.1, inertia=wp.mat33(np.eye(3) * 1e-4), label=CAMERA if k == 2 else f"/vehicle/link_{k}"
        )
        chain.append(builder.add_joint_revolute(parent, link, axis=axis, label=joint))
        parent = link
    builder.add_articulation(chain)
    return builder.finalize()


@pytest.mark.parametrize(
    ("bad", "attribute"),
    [
        ({"attitude_gain": -1.0}, "nexus:attitudeGain"),
        ({"rate_limit": 0.0}, "nexus:rateLimit"),
        ({"rate_limit": -1.0}, "nexus:rateLimit"),
        ({"rate_gain": (0.5, -0.1, 0.5)}, "nexus:rateGain"),
        ({"integral_gain": (0.0, 0.0, -2.0)}, "nexus:integralGain"),
    ],
)
def test_a_gain_below_zero_or_a_rate_limit_at_or_below_zero_fails_and_names_the_prim_and_the_attribute(bad, attribute):
    """A gain below 0, or a rate limit at or below 0, fails the build, the message naming the prim and the
    attribute.

    Given a vehicle's square chain and each bad value in turn, an attitude gain of −1, a rate limit of 0 and of −1,
    a rate gain of −0.1 on one motor and an integral gain of −2 on one motor, when the test makes the controller, then it
    raises `ValueError`, and the message names the controller's prim and the attribute.
    """
    values = {
        "attitude_gain": ATTITUDE_GAIN,
        "rate_limit": 2.0,
        "rate_gain": RATE_GAIN,
        "integral_gain": (0.0, 0.0, 0.0),
        **bad,
    }

    try:
        GimbalController(model=_vehicle(), path=PRIM, inputs={"camera": [CAMERA], "joints": JOINTS}, **values)
        message = "no error"
    except ValueError as exc:
        message = str(exc)

    assert [s in message for s in (f"{PRIM}:", attribute)] == [True, True], message
