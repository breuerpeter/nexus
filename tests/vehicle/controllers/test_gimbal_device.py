"""A gimbal device's side of its MAVLink link: what a datagram means, and what the device sends.

Each test hands the device the datagrams a gimbal manager sends, packed by pymavlink as the autopilot of system 3,
and reads the setpoint and the reply datagrams it gives back. No test opens a socket. An attitude is a camera's
frame against North, East, Down, with x along the view, y right and z down, so a pitch of −30° looks 30° down.
"""

import math

import newton
import numpy as np
import pytest
import warp as wp
from pymavlink.dialects.v20 import common as mavlink

from nexus_uav._src.vehicle.controllers.gimbal import axis_limits
from nexus_uav._src.vehicle.controllers.gimbal_device import GimbalDevice

pytestmark = pytest.mark.usefixtures("warp_cpu")

SYSTEM = 3  # the manager's MAVLink system
GIMBAL = 154  # the component id of a gimbal device, MAV_COMP_ID_GIMBAL
ROLL_LOCK, PITCH_LOCK, YAW_LOCK = 4, 8, 16  # GIMBAL_DEVICE_FLAGS
IN_VEHICLE, IN_EARTH, ACCEPTS_EARTH = 32, 64, 128
NAN = math.nan
DOWN_30_WXYZ = (0.9659258, 0.0, -0.2588190, 0.0)  # a pitch of −30°, as MAVLink orders a quaternion
DOWN_30 = (0.0, -0.2588190, 0.0, 0.9659258)  # the same, `(x, y, z, w)`
RATE = (0.1, 0.2, 0.3)  # rad/s
REQUEST_MESSAGE, DO_SET_MODE = 512, 176  # MAV_CMD
INFORMATION, ATTITUDE = 283, 30  # message ids
ACCEPTED, UNSUPPORTED = 0, 3  # MAV_RESULT


def _pack(message) -> bytes:
    """`message` as a datagram from the manager, the autopilot of system 3."""
    return message.pack(mavlink.MAVLink(None, srcSystem=SYSTEM, srcComponent=1))


def _set_attitude(q, rate, flags: int, component: int = GIMBAL) -> bytes:
    """A `GIMBAL_DEVICE_SET_ATTITUDE` for `component`: `q` as `(w, x, y, z)`, `rate` [rad/s] and `flags`."""
    return _pack(mavlink.MAVLink_gimbal_device_set_attitude_message(SYSTEM, component, flags, list(q), *rate))


def _command(command: int, param1: float = 0.0, component: int = GIMBAL) -> bytes:
    """A `COMMAND_LONG` for `component` of every system."""
    return _pack(mavlink.MAVLink_command_long_message(0, component, command, 0, param1, 0, 0, 0, 0, 0, 0))


def _decode(datagrams) -> list:
    """The MAVLink messages in `datagrams`, in order."""
    return mavlink.MAVLink(None).parse_buffer(b"".join(datagrams)) or []


def _row(setpoint) -> list[float]:
    """A setpoint's attitude, rate and lock bit as one row of numbers; a missing setpoint as a row of infinity."""
    if setpoint is None:
        return [math.inf] * 8
    return [*setpoint.attitude, *setpoint.rate, float(setpoint.locked)]


def _only(setpoints):
    """The one setpoint among `setpoints`, what a datagram of one message gives; `None` for none or more."""
    return setpoints[0] if len(setpoints) == 1 else None


def _square_chain(limits):
    """A pan joint about z, a roll joint about x and a tilt joint about y on a fixed base, as the fixture quad's
    gimbal has them, each with its lowest and highest angle [rad] from `limits`, `None` for no limit. Returns the
    model and the three joints.
    """
    builder = newton.ModelBuilder()
    joints, parent = [], -1
    for axis, limit in zip(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), limits, strict=True):
        link = builder.add_link(mass=0.1, inertia=wp.mat33(np.eye(3) * 1e-4))
        lower, upper = limit if limit is not None else (None, None)
        joints.append(builder.add_joint_revolute(parent, link, axis=axis, limit_lower=lower, limit_upper=upper))
        parent = link
    builder.add_articulation(joints)
    return builder.finalize(), joints


def test_each_set_attitude_gives_the_laws_whole_setpoint():
    """Each `GIMBAL_DEVICE_SET_ATTITUDE` gives the law's whole setpoint: the attitude, the rate and the yaw lock
    bit, a NaN attitude as NaN and a NaN rate as no rate, whatever the roll and pitch lock flags say.

    Given, in turn, a `q` of 30° down with a rate of (0.1, 0.2, 0.3) and `YAW_LOCK` with `YAW_IN_EARTH_FRAME`; a
    NaN `q` with that rate and `YAW_IN_VEHICLE_FRAME`; that `q` with NaN rates; and the first again with
    `ROLL_LOCK` and `PITCH_LOCK` cleared, when the device handles each, then the setpoint is the attitude as
    `(x, y, z, w)` with the rate and the lock; a NaN attitude with the rate and no lock; the attitude with a rate
    of 0; and the first setpoint again.
    """
    device = GimbalDevice()
    locks = ROLL_LOCK | PITCH_LOCK
    datagrams = [
        _set_attitude(DOWN_30_WXYZ, RATE, locks | YAW_LOCK | IN_EARTH),
        _set_attitude((NAN,) * 4, RATE, locks | IN_VEHICLE),
        _set_attitude(DOWN_30_WXYZ, (NAN,) * 3, locks | YAW_LOCK | IN_EARTH),
        _set_attitude(DOWN_30_WXYZ, RATE, YAW_LOCK | IN_EARTH),
    ]

    rows = [_row(_only(device.handle(datagram)[0])) for datagram in datagrams]

    expected = [
        [*DOWN_30, *RATE, 1.0],
        [NAN, NAN, NAN, NAN, *RATE, 0.0],
        [*DOWN_30, 0.0, 0.0, 0.0, 1.0],
        [*DOWN_30, *RATE, 1.0],
    ]
    np.testing.assert_allclose(rows, expected, atol=1e-6, equal_nan=True)


def test_two_setpoints_in_one_datagram_both_reach_the_law_in_their_order():
    """Each `GIMBAL_DEVICE_SET_ATTITUDE` gives a setpoint, also when one datagram holds two: a write is an event,
    so the second doesn't replace the first.

    Given one datagram that holds a `q` of 30° down with no rate and then a NaN `q` with a rate of
    (0.1, 0.2, 0.3), both with `YAW_LOCK` and `YAW_IN_EARTH_FRAME`, when the device handles it, then it gives the
    attitude with a rate of 0 and then a NaN attitude with the rate, in that order.
    """
    device = GimbalDevice()
    flags = YAW_LOCK | IN_EARTH
    datagram = _set_attitude(DOWN_30_WXYZ, (NAN,) * 3, flags) + _set_attitude((NAN,) * 4, RATE, flags)

    rows = [_row(setpoint) for setpoint in device.handle(datagram)[0]]

    expected = [[*DOWN_30, 0.0, 0.0, 0.0, 1.0], [NAN, NAN, NAN, NAN, *RATE, 1.0]]
    np.testing.assert_equal(np.round(rows, 6).tolist() if len(rows) == 2 else rows, np.round(expected, 6).tolist())


def test_the_yaw_frame_flags_name_the_frame_of_the_attitudes_heading():
    """The yaw frame flags name the frame of the attitude's heading, and with neither flag or both the lock bit
    names it.

    Given a `q` with, in turn, `YAW_LOCK` and `YAW_IN_VEHICLE_FRAME`; no lock and `YAW_IN_EARTH_FRAME`;
    `YAW_LOCK` alone; no flag; and `YAW_LOCK` with both frame flags, when the device handles each, then the lock bit
    follows `YAW_LOCK` and the attitude's frame is vehicle, earth, earth, vehicle and earth.
    """
    device = GimbalDevice()
    flags = [YAW_LOCK | IN_VEHICLE, IN_EARTH, YAW_LOCK, 0, YAW_LOCK | IN_VEHICLE | IN_EARTH]

    setpoints = [_only(device.handle(_set_attitude(DOWN_30_WXYZ, (0.0, 0.0, 0.0), flag))[0]) for flag in flags]

    frames = [(s.locked, s.earth) if s is not None else None for s in setpoints]
    assert frames == [(True, False), (False, True), (True, True), (False, False), (True, True)]


def test_a_request_for_the_devices_information_gets_an_ack_and_the_information():
    """A request for `GIMBAL_DEVICE_INFORMATION`, to the device or to every component, gets `COMMAND_ACK` accepted
    and the information: each axis's limits from its joint, and what the gimbal can do.

    Given the fixture's chain as a Newton model with limits of −90° to 30° on tilt, ±45° on roll and none on pan,
    when a `COMMAND_LONG` `MAV_CMD_REQUEST_MESSAGE` for it arrives for component 0 and then for 154, then each
    reply is a `COMMAND_ACK` `MAV_RESULT_ACCEPTED` and a `GIMBAL_DEVICE_INFORMATION` with those pitch and roll
    limits, NaN yaw limits with `SUPPORTS_INFINITE_YAW`, and capability flags for three axes, the roll and pitch
    locks, yaw follow, yaw lock and a yaw in the earth frame, and no other.
    """
    quarter, sixth, half = math.pi / 4.0, math.pi / 6.0, math.pi / 2.0
    model, joints = _square_chain([None, (-quarter, quarter), (-half, sixth)])
    device = GimbalDevice(**axis_limits(model, joints))

    replies = [
        _decode(device.handle(_command(REQUEST_MESSAGE, INFORMATION, component))[1]) for component in (0, GIMBAL)
    ]

    seen = []
    for reply in replies:
        kinds = [message.get_type() for message in reply]
        if kinds != ["COMMAND_ACK", "GIMBAL_DEVICE_INFORMATION"]:
            seen.append(kinds)
            continue
        ack, information = reply
        limits = [getattr(information, f"{axis}_{end}") for axis in ("roll", "pitch", "yaw") for end in ("min", "max")]
        seen.append((ack.command, ack.result, np.round(limits, 5).tolist(), information.cap_flags))
    # Roll, pitch and yaw axes, 4 + 32 + 256; the roll and pitch locks, 16 + 128; yaw follow and yaw lock,
    # 512 + 1024; an endless yaw, 2048; and a yaw in the earth frame, 4096.
    answer = (REQUEST_MESSAGE, ACCEPTED, [-0.7854, 0.7854, -1.5708, 0.5236, NAN, NAN], 8116)
    np.testing.assert_equal(seen, [answer, answer])


def test_a_command_the_device_does_not_serve_is_unsupported_only_when_it_names_the_device():
    """A command the device doesn't serve gets `COMMAND_ACK` unsupported when it names the device, and no answer
    when it names every component or another one.

    Given a request for `ATTITUDE` and a `MAV_CMD_DO_SET_MODE`, each for component 154, then the second for
    component 0 and for component 1, when the device handles each, then the first two reply `COMMAND_ACK`
    `MAV_RESULT_UNSUPPORTED` and the last two reply nothing.
    """
    device = GimbalDevice()
    datagrams = [
        _command(REQUEST_MESSAGE, ATTITUDE),
        _command(DO_SET_MODE),
        _command(DO_SET_MODE, component=0),
        _command(DO_SET_MODE, component=1),
    ]

    replies = [_decode(device.handle(datagram)[1]) for datagram in datagrams]

    answers = [[(m.get_type(), m.command, m.result) for m in reply] for reply in replies]
    unsupported = [("COMMAND_ACK", REQUEST_MESSAGE, UNSUPPORTED)], [("COMMAND_ACK", DO_SET_MODE, UNSUPPORTED)]
    assert answers == [*unsupported, [], []]


def test_a_request_whose_message_id_is_no_number_is_a_command_the_device_does_not_serve():
    """A request for a message whose id is NaN or infinite is a command the device doesn't serve: it raises
    nothing.

    Given a `MAV_CMD_REQUEST_MESSAGE` for component 154 with a `param1` of, in turn, NaN, infinity and minus
    infinity, when the device handles each, then each reply is one `COMMAND_ACK` `MAV_RESULT_UNSUPPORTED`.
    """
    device = GimbalDevice()

    answers = []
    for param1 in (NAN, math.inf, -math.inf):
        try:
            replies = _decode(device.handle(_command(REQUEST_MESSAGE, param1))[1])
            answers.append([(m.get_type(), m.result) for m in replies])
        except Exception as exc:
            answers.append(exc)

    assert answers == [[("COMMAND_ACK", UNSUPPORTED)]] * 3


def test_a_datagram_the_device_does_not_act_on_gives_no_setpoint_and_no_reply():
    """A datagram the device doesn't act on gives no setpoint and no reply.

    Given a device that gives a setpoint for a `GIMBAL_DEVICE_SET_ATTITUDE` of its own, and then, in turn, an
    `AUTOPILOT_STATE_FOR_GIMBAL_DEVICE`, a `HEARTBEAT`, an `ATTITUDE`, a `GIMBAL_DEVICE_SET_ATTITUDE` for
    component 1, and bytes that are no MAVLink, when the device handles each, then it gives no setpoint, no reply and no
    error.
    """
    device = GimbalDevice()
    state = mavlink.MAVLink_autopilot_state_for_gimbal_device_message(
        SYSTEM, GIMBAL, 1000, [1.0, 0.0, 0.0, 0.0], 0, 0.0, 0.0, 0.0, 0, 0.1, 0, 0
    )
    heartbeat = mavlink.MAVLink_heartbeat_message(mavlink.MAV_TYPE_QUADROTOR, mavlink.MAV_AUTOPILOT_PX4, 0, 0, 0, 3)
    attitude = mavlink.MAVLink_attitude_message(1000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    other = _set_attitude(DOWN_30_WXYZ, RATE, YAW_LOCK | IN_EARTH, component=1)
    ignored = [_pack(state), _pack(heartbeat), _pack(attitude), other, b"\xfdnot MAVLink at all"]

    own = _only(device.handle(_set_attitude(DOWN_30_WXYZ, RATE, YAW_LOCK | IN_EARTH))[0])
    outcomes = []
    for datagram in ignored:
        try:
            outcomes.append(device.handle(datagram))
        except Exception as exc:
            outcomes.append(exc)

    assert (own is not None, outcomes) == (True, [([], [])] * 5)


def test_only_a_message_moves_the_address_the_device_answers():
    """The device answers the address its last message came from, and bytes that are no MAVLink move it nowhere.

    Given a device whose manager spoke from one address, when bytes that are no MAVLink and an empty datagram
    arrive from a second address, then the device still answers the first; and when a `HEARTBEAT` then arrives
    from the second, the device answers the second.
    """
    device = GimbalDevice()
    first, second = ("127.0.0.1", 50001), ("127.0.0.1", 50002)
    beat = _pack(mavlink.MAVLink_heartbeat_message(mavlink.MAV_TYPE_QUADROTOR, mavlink.MAV_AUTOPILOT_PX4, 0, 0, 0, 3))

    device.handle(beat, sender=first)
    device.handle(b"\xfdnot MAVLink at all", sender=second)
    device.handle(b"", sender=second)
    kept = device.manager
    device.handle(beat, sender=second)

    assert (kept, device.manager) == (first, second)


def test_the_status_carries_the_cameras_attitude_its_rate_and_the_locks_in_force():
    """The status carries the camera's attitude against North, East, Down and its rate, with the roll and pitch
    locks set, the yaw lock in force, and the yaw frame named as earth and accepted.

    Given a camera 30° down that faces East with a rate of 0.1 rad/s about its y axis, and yaw locked, when the
    device builds the status, then its `q` is that attitude as `(w, x, y, z)`, its rate that rate, its flags `ROLL_LOCK`,
    `PITCH_LOCK`, `YAW_LOCK`, `YAW_IN_EARTH_FRAME` and `ACCEPTS_YAW_IN_EARTH_FRAME`, its `delta_yaw` NaN and its
    failure flags 0; and with yaw unlocked the flags drop `YAW_LOCK` alone.
    """
    device = GimbalDevice()
    east_down_30 = (0.1830127, -0.1830127, 0.6830127, 0.6830127)  # a yaw of 90°, then a pitch of −30°
    rate = (0.0, 0.1, 0.0)

    statuses = []
    for flags in (YAW_LOCK | IN_EARTH, IN_VEHICLE):
        device.handle(_set_attitude((NAN,) * 4, (0.0, 0.0, 0.0), flags))
        statuses += _decode([device.status(1500, east_down_30, rate)]) or [None]

    seen = []
    for status in statuses:
        if status is None or status.get_type() != "GIMBAL_DEVICE_ATTITUDE_STATUS":
            seen.append(None)
            continue
        rates = [status.angular_velocity_x, status.angular_velocity_y, status.angular_velocity_z]
        numbers = np.round([*status.q, *rates, status.delta_yaw], 5).tolist()
        seen.append((status.time_boot_ms, numbers, status.flags, status.failure_flags))
    numbers = [0.68301, 0.18301, -0.18301, 0.68301, 0.0, 0.1, 0.0, NAN]
    locks = ROLL_LOCK | PITCH_LOCK | IN_EARTH | ACCEPTS_EARTH
    np.testing.assert_equal(seen, [(1500, numbers, locks | YAW_LOCK, 0), (1500, numbers, locks, 0)])
