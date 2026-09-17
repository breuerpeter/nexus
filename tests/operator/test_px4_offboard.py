"""Px4Offboard command encoding against a fake mavutil connection, with no real PX4.

The tests bypass the background pump thread, setting _mav/_sysid/_compid directly, and assert the
MAVLink messages the operator verbs emit; the flight itself is the integration proof.

The verbs are **requests**: they lodge intent and return, and the pump puts the commands on
the wire. So a test that wants to see bytes drives ``_service_requests`` itself, with an
explicit ``now``, instead of waiting on a thread.
"""

import json
import math
import time

import pytest

from nexus._src.operator import px4_offboard as px4mod
from nexus._src.operator.qgc_plan import MissionItem


class _FakeMav:
    def __init__(self):
        self.calls = []

    def command_long_send(self, *a):
        self.calls.append(("command_long", a))

    def manual_control_send(self, *a):
        self.calls.append(("manual_control", a))

    def command_int_send(self, *a):
        self.calls.append(("command_int", a))

    def param_set_send(self, *a):
        self.calls.append(("param_set", a))

    def heartbeat_send(self, *a):
        self.calls.append(("heartbeat", a))

    def mission_count_send(self, *a):
        self.calls.append(("mission_count", a))

    def mission_item_int_send(self, *a):
        self.calls.append(("mission_item_int", a))


class _FakeConn:
    def __init__(self):
        self.mav = _FakeMav()


def _pilot():
    p = px4mod.Px4Offboard()
    p._mav = _FakeConn()
    p._sysid, p._compid = 1, 1
    return p


def _service(p, now=None, state=None):
    """Run one pump service pass at ``now``, which defaults to real monotonic, the clock
    ``arm()`` stamps ``_arm_since`` with.
    """
    state = state if state is not None else px4mod._RequestState()
    p._service_requests(time.monotonic() if now is None else now, state)
    return state


def _sent(p, command):
    from pymavlink import mavutil

    _ = mavutil
    return [a for c, a in p._mav.mav.calls if c == "command_long" and a[2] == command]


def test_set_mode_position_sends_do_set_mode_main3():
    from pymavlink import mavutil

    p = _pilot()
    p.set_mode("Position")
    assert p._want_mode == "Position"  # the verb only lodges the request …
    assert not p._mav.mav.calls
    _service(p)  # … the pump puts it on the wire
    cmd, args = p._mav.mav.calls[-1]
    assert cmd == "command_long"
    # args: (sysid, compid, command, confirmation, p1..p7)
    assert args[2] == mavutil.mavlink.MAV_CMD_DO_SET_MODE
    assert args[4] == mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED  # p1 base_mode
    assert args[5] == 3  # p2 = PX4 main mode POSCTL


def test_mode_request_clears_once_telemetry_confirms_it():
    # The pump must stop commanding a confirmed mode: PX4 leaves Takeoff mode for Hold by itself
    # on arrival, and a latched request would fight it all the way back.
    from pymavlink import mavutil

    p = _pilot()
    p.set_mode("Takeoff")
    st = _service(p, now=100.0)
    assert len(_sent(p, mavutil.mavlink.MAV_CMD_DO_SET_MODE)) == 1
    p._mode = "Takeoff"  # telemetry confirms
    _service(p, now=200.0, state=st)
    assert p._want_mode is None
    assert len(_sent(p, mavutil.mavlink.MAV_CMD_DO_SET_MODE)) == 1  # no further sends


def test_arm_waits_for_the_mode_change_and_the_settle_period():
    # The arm ordering: PX4 confirms the mode command for Takeoff mode *before* the arm command, always. PX4's
    # arming checks are mode-dependent, and PX4 denies arming during the Extended Kalman Filter (EKF)
    # convergence transient.
    from pymavlink import mavutil

    arm_cmd = mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
    p = _pilot()
    p._rel_alt = 0.0  # PX4 is publishing a position estimate
    p.takeoff(5.0)
    t0 = p._arm_since
    st = _service(p, now=t0, state=None)
    assert not _sent(p, arm_cmd)  # settle period hasn't elapsed
    _service(p, now=t0 + 10.0, state=st)
    assert not _sent(p, arm_cmd)  # settled, but the mode is still unconfirmed
    p._mode = "Takeoff"
    _service(p, now=t0 + 20.0, state=st)
    assert len(_sent(p, arm_cmd)) == 1


def test_arm_request_clears_once_armed_so_the_pump_never_re_arms():
    # PX4 disarms itself after landing. A latched arm request would put the vehicle straight
    # back into the air.
    from pymavlink import mavutil

    arm_cmd = mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
    p = _pilot()
    p._rel_alt = 0.0
    p.arm()
    t0 = p._arm_since
    st = _service(p, now=t0 + 10.0)
    assert len(_sent(p, arm_cmd)) == 1
    p._armed = True
    _service(p, now=t0 + 20.0, state=st)
    assert p._want_arm is False
    p._armed = False  # PX4's own post-landing disarm
    _service(p, now=t0 + 30.0, state=st)
    assert len(_sent(p, arm_cmd)) == 1  # not re-armed


# ---- the stick-gesture arm, the manual path a real pilot flies -----------------------------------


def _gesturing(p):
    """A pilot mid-manual-flight: Altitude confirmed, a position estimate, a gesture arm lodged."""
    p.set_mode("Altitude")
    p._mode = "Altitude"  # telemetry confirms
    p._rel_alt = 0.0  # PX4 is publishing a position estimate
    p.arm(gesture=True)
    return p


def test_gesture_arm_holds_neutral_until_armable():
    # PX4 arms on the RISING edge of the gesture, so presenting it before the vehicle can act on it
    # burns the one edge there is. Until armable the pump streams neutral sticks and nothing else.
    from pymavlink import mavutil

    p = _pilot()
    p.arm(gesture=True)
    assert p._rc == (0, 0, 500, 0)  # a gesture arm needs a live manual-control source; arm starts one
    st = _service(p, now=p._arm_since + 10.0)  # settled, but _rel_alt is still None
    assert p._rc == (0, 0, 500, 0)
    assert not _sent(p, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)  # never the command path
    assert (st.gesture_held, st.gesture_since) == (False, 0.0)


def test_gesture_arm_drives_the_gesture_once_armable():
    from pymavlink import mavutil

    p = _gesturing(_pilot())
    t0 = p._arm_since
    _service(p, now=t0)  # the settle window hasn't elapsed
    assert p._rc == (0, 0, 500, 0)
    _service(p, now=t0 + 10.0)
    assert p._rc == (0, 0, 0, 1000)  # throttle fully down + yaw fully right
    assert not _sent(p, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)


def test_gesture_arm_recreates_the_edge():
    # A gesture PX4 denies never re-fires until the sticks leave it and come back, so a pump that
    # simply held the gesture would wait out the whole flight on one dead edge.
    p = _gesturing(_pilot())
    t0 = p._arm_since
    st = _service(p, now=t0 + 10.0)
    assert p._rc == (0, 0, 0, 1000)
    _service(p, now=t0 + 11.0, state=st)
    assert p._rc == (0, 0, 0, 1000)  # still inside the hold: the gesture stays put
    _service(p, now=t0 + 10.0 + px4mod._GESTURE_HOLD_S, state=st)
    assert p._rc == (0, 0, 500, 0)  # the falling edge
    _service(p, now=t0 + 10.0 + px4mod._GESTURE_HOLD_S + px4mod._GESTURE_GAP_S, state=st)
    assert p._rc == (0, 0, 0, 1000)  # and the next rising one


def test_gesture_arm_restores_neutral_on_confirmation():
    # Leaving the sticks in the gesture after arming would fly the vehicle at throttle-down and
    # full-right yaw the instant the motors spin.
    p = _gesturing(_pilot())
    st = _service(p, now=p._arm_since + 10.0)
    assert p._rc == (0, 0, 0, 1000)
    p._armed = True  # telemetry confirms
    _service(p, now=p._arm_since + 20.0, state=st)
    assert (p._want_arm, p._arm_gesture) == (False, False)
    assert p._rc == (0, 0, 500, 0)


def test_takeoff_still_uses_the_command_arm():
    # The gesture path must leave the autonomous profile untouched: arm() defaults to the command,
    # and an auto-mode flight puts no MANUAL_CONTROL on the wire at all.
    from pymavlink import mavutil

    p = _pilot()
    p._rel_alt = 0.0
    p.takeoff(5.0)
    p._mode = "Takeoff"
    st = _service(p, now=p._arm_since + 10.0)
    assert len(_sent(p, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)) == 1
    assert p._arm_gesture is False
    assert p._rc is None  # no stick stream was ever started
    assert (st.gesture_held, st.gesture_since) == (False, 0.0)


def test_disarm_drops_a_pending_gesture():
    p = _gesturing(_pilot())
    p.disarm()
    assert (p._want_arm, p._arm_gesture) == (False, False)


def test_set_mode_unknown_raises():
    with pytest.raises(ValueError):
        _pilot().set_mode("Loiterish")


def test_set_rc_scales_to_px4_units():
    p = _pilot()
    p.set_rc(roll=0.0, pitch=0.0, throttle=1.0, yaw=0.0)
    assert p._rc == (0, 0, 1000, 0)  # (x=pitch, y=roll, z=throttle, r=yaw)
    p.set_rc(throttle=0.5, roll=-1.0)
    assert p._rc == (0, -1000, 500, 0)


def test_disarm_sends_arm_disarm_zero():
    from pymavlink import mavutil

    p = _pilot()
    p.disarm()
    _cmd, args = p._mav.mav.calls[-1]
    assert args[2] == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
    assert args[4] == 0.0  # p1 = 0 -> disarm


def test_param_set_stays_float():
    from pymavlink import mavutil

    p = _pilot()
    p.param_set("MPC_THR_HOVER", 0.42)
    cmd, args = p._mav.mav.calls[-1]
    assert cmd == "param_set"
    assert args[2] == b"MPC_THR_HOVER"
    assert args[3] == pytest.approx(0.42)
    assert args[4] == mavutil.mavlink.MAV_PARAM_TYPE_REAL32


def test_param_set_int_sends_the_bit_pattern():
    # PARAM_SET has one float field and PX4 reads an int parameter out of its *bits*. Sending
    # float(1) instead puts 1.0 on the wire, which PX4 reads as 1065353216, so COM_RC_IN_MODE
    # would silently land on a nonsense value and the only symptom is a gesture that never arms.
    import struct

    from pymavlink import mavutil

    p = _pilot()
    p.param_set_int("COM_RC_IN_MODE", 1)
    cmd, args = p._mav.mav.calls[-1]
    assert cmd == "param_set"
    assert args[2] == b"COM_RC_IN_MODE"
    assert args[3] == struct.unpack("<f", struct.pack("<i", 1))[0]
    assert args[4] == mavutil.mavlink.MAV_PARAM_TYPE_INT32


def test_decode_mode_roundtrip():
    from pymavlink import mavutil

    # POSCTL custom_mode: main=3 in byte 2
    custom = 3 << 16
    assert px4mod.Px4Offboard._decode_mode(custom) == "Position"
    # Land mode: main=4, sub=6
    custom = (4 << 16) | (6 << 24)
    assert px4mod.Px4Offboard._decode_mode(custom) == "Land"
    _ = mavutil  # keep import used


def test_takeoff_forwards_mis_takeoff_alt_and_sets_mode():
    from pymavlink import mavutil

    p = _pilot()
    p.takeoff(7.0)
    sends = p._mav.mav.calls
    # MIS_TAKEOFF_ALT param forwarded
    assert any(c == "param_set" and a[2] == b"MIS_TAKEOFF_ALT" and a[3] == 7.0 for c, a in sends)
    # Takeoff mode + arm requested, in that order: takeoff() owns the sequence
    assert (p._want_mode, p._want_arm) == ("Takeoff", True)
    _service(p)
    assert _sent(p, mavutil.mavlink.MAV_CMD_DO_SET_MODE)


def test_import_nexus_pins_common_mavlink_dialect():
    # Regression, caught only by a live flight: `import nexus` eagerly imports Px4Offboard,
    # which can be the *first* pymavlink import. If it doesn't pin the common dialect, the
    # Hardware In The Loop (HIL) controller's later dialect setting is a cached no-op and HIL_GPS
    # loses its `id`/`yaw` fields, crashing the lockstep loop. Run in a fresh process so import order
    # is the real one.
    import subprocess
    import sys

    code = (
        "import nexus; from pymavlink import mavutil; import inspect; "
        "p = list(inspect.signature(mavutil.mavlink.MAVLink(0).hil_gps_send).parameters); "
        "assert 'id' in p and 'yaw' in p, p"
    )
    r = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_set_mode_manual_starts_manual_control_stream():
    # Regression, caught by a live flight: PX4 rejects a switch into a manual mode unless a
    # manual-control source is live, so set_mode must auto-start a neutral MANUAL_CONTROL stream
    # first, else PX4 silently ignores the switch and the vehicle stays in, for example, Hold.
    from pymavlink import mavutil

    p = _pilot()
    assert p._rc is None
    p.set_mode("Position")
    assert p._rc == (0, 0, 500, 0)  # neutral: centred sticks, throttle 0.5
    _service(p)
    assert _sent(p, mavutil.mavlink.MAV_CMD_DO_SET_MODE)


def test_set_mode_auto_does_not_start_stream():
    p = _pilot()
    p.set_mode("Hold")  # an auto mode needs no joystick: must not start a manual-control stream
    assert p._rc is None


# ---- the local tangent plane: world axes -> WGS84, and arrival ----------------------------------

LAT0, LON0 = 47.397742, 8.545594  # PX4's default Software In The Loop (SITL) origin
AMSL0, REL_ALT0 = 488.0, 5.0  # so home_amsl = 483.0
M_PER_DEG = math.pi / 180.0 * 6378137.0


def _wrap_deg(deg):
    """Wrap degrees to (-180, 180], the range _wrap_pi puts the commanded heading in."""
    return (deg + 180.0) % 360.0 - 180.0


def _flying(p, *, lat=LAT0, lon=LON0, rel_alt=REL_ALT0, yaw=None):
    """Fill the telemetry cache as the pump would from GLOBAL_POSITION_INT + ATTITUDE."""
    p._lat, p._lon, p._rel_alt = lat, lon, rel_alt
    p._amsl = AMSL0 - REL_ALT0 + rel_alt  # home sea-level altitude stays fixed; the vehicle's rides its rel alt
    p._yaw = yaw
    return p


def _last_goto(p):
    """The (lat, lon, alt_amsl, yaw) of the last DO_REPOSITION put on the wire."""
    from pymavlink import mavutil

    sends = [a for c, a in p._mav.mav.calls if c == "command_int" and a[3] == mavutil.mavlink.MAV_CMD_DO_REPOSITION]
    a = sends[-1]
    # (sysid, compid, frame, command, current, autocontinue, p1..p4, x, y, z)
    return a[10] / 1e7, a[11] / 1e7, a[12], a[9]


def test_goto_maps_plus_x_north_and_plus_y_west():
    # Newton's world frame is Forward Left Up (FLU) with north along +x, which forces east onto -y,
    # see #61. Getting this backwards flies a mirrored path that still "arrives" at every waypoint.
    # Tolerances are 2 cm: DO_REPOSITION carries the fix as 1e-7 deg integers, about 1.1 cm, so that
    # quantisation, not the conversion, is the floor here.
    p = _flying(_pilot())
    p.goto((100.0, 0.0, 5.0))
    lat, lon, _, _ = _last_goto(p)
    assert (lat - LAT0) * M_PER_DEG == pytest.approx(100.0, abs=0.02)  # +x is north
    assert lon == pytest.approx(LON0, abs=1e-7)

    p.goto((0.0, 100.0, 5.0))
    lat, lon, _, _ = _last_goto(p)
    assert lat == pytest.approx(LAT0, abs=1e-7)
    east = (lon - LON0) * M_PER_DEG * math.cos(math.radians(LAT0))
    assert east == pytest.approx(-100.0, abs=0.02)  # +y is *west*


def test_goto_resolves_altitude_against_home_amsl():
    p = _flying(_pilot())
    p.goto((0.0, 0.0, 12.0))
    _, _, alt_amsl, _ = _last_goto(p)
    assert alt_amsl == pytest.approx(AMSL0 - REL_ALT0 + 12.0)  # home_amsl + z


def test_goto_anchor_is_captured_once_and_needs_a_position():
    p = _pilot()
    with pytest.raises(RuntimeError):
        p.goto((1.0, 2.0, 3.0))  # no GLOBAL_POSITION_INT yet
    _flying(p)
    p.goto((0.0, 0.0, 5.0))
    anchor = p._anchor
    _flying(p, lat=LAT0 + 0.01, lon=LON0 + 0.01, rel_alt=9.0)  # the vehicle moves on
    p.goto((0.0, 0.0, 5.0))
    assert p._anchor == anchor  # the anchor doesn't drift with it
    lat, lon, _, _ = _last_goto(p)
    assert (lat, lon) == pytest.approx((LAT0, LON0), abs=1e-9)


def test_goto_yaw_overrides_the_goals_own_yaw():
    from nexus._src.core.schema import PositionGoal

    p = _flying(_pilot())
    p.goto(PositionGoal(pos=(0.0, 0.0, 5.0), yaw=1.0))
    assert _last_goto(p)[3] == pytest.approx(-1.0)  # world -> PX4 reflection
    p.goto(PositionGoal(pos=(0.0, 0.0, 5.0), yaw=1.0), yaw=2.0)
    assert _last_goto(p)[3] == pytest.approx(-2.0)
    p.goto((0.0, 0.0, 5.0))
    assert math.isnan(_last_goto(p)[3])  # no yaw commanded -> PX4 keeps its own


def test_goto_yaw_is_reflected_from_world_to_px4_frame():
    # The yaw twin of the +y/west position test. World yaw is right-handed about +z, so +90 deg
    # turns +x, north, toward +y, *west*; PX4's North East Down (NED) heading is positive north toward
    # *east*. Sending a world yaw through unconverted faces the vehicle at the mirror-image heading, and
    # because at_target would then compare two PX4-frame angles, it would report success while doing it.
    p = _flying(_pilot())
    for world_deg, want_px4_deg in ((90.0, -90.0), (-90.0, 90.0), (0.0, 0.0), (45.0, -45.0)):
        p.goto((0.0, 0.0, 5.0), yaw=math.radians(world_deg))
        assert math.degrees(_last_goto(p)[3]) == pytest.approx(want_px4_deg, abs=1e-9)
    # 180 deg is its own reflection. _wrap_pi lands it on -180, the same heading in the other
    # representation, which is why every comparison against it goes through the wrap.
    p.goto((0.0, 0.0, 5.0), yaw=math.radians(180.0))
    assert abs(math.degrees(_last_goto(p)[3])) == pytest.approx(180.0, abs=1e-9)


def test_flight_yaw_sweep_reaches_px4_as_the_old_compass_headings():
    # The px4_sitl profile's sweep is compass, NED, and flight.py negates on the way in. Pin that
    # the bytes PX4 gets are the old ones, in the old order: the gate doesn't check yaw, so
    # nothing else would catch a reversed sweep.
    p = _flying(_pilot())
    for heading in (90.0, 180.0, 270.0, 0.0):
        p.goto((0.0, 0.0, 5.0), yaw=-math.radians(heading))
        sent = math.degrees(_last_goto(p)[3])
        assert sent == pytest.approx(_wrap_deg(heading), abs=1e-9)


def test_at_target_is_false_with_nothing_commanded():
    assert _flying(_pilot()).at_target() is False


def test_at_target_for_the_climb():
    p = _flying(_pilot(), rel_alt=0.0)
    p.takeoff(5.0)
    assert p.at_target() is False
    _flying(p, rel_alt=3.5)
    assert p.at_target() is False  # 1.5 m short, outside the 1.0 m tolerance
    _flying(p, rel_alt=4.5)
    assert p.at_target() is True


def test_at_target_for_a_position():
    p = _flying(_pilot())
    p.goto((50.0, 0.0, 5.0))
    assert p.at_target() is False
    _flying(p, lat=LAT0 + math.degrees(47.0 / 6378137.0), rel_alt=5.0)
    assert p.at_target() is False  # 3 m out, outside the 2.0 m arrival radius
    _flying(p, lat=LAT0 + math.degrees(49.0 / 6378137.0), rel_alt=5.0)
    assert p.at_target() is True
    _flying(p, lat=LAT0 + math.degrees(49.0 / 6378137.0), rel_alt=8.0)
    assert p.at_target() is False  # arrived laterally but 3 m high: the radius is 3D


def test_at_target_for_a_yaw():
    # goto takes *world* yaw; _flying fills _yaw as ATTITUDE does, in PX4's frame. World +180 is
    # PX4 +180, its own reflection, so this leg reads the same in both.
    p = _flying(_pilot(), yaw=0.0)
    p.goto((0.0, 0.0, 5.0), yaw=math.radians(180.0))
    assert p.at_target() is False  # in position, wrong heading
    _flying(p, yaw=math.radians(170.0))
    assert p.at_target() is False  # 10 deg out, outside the 5 deg tolerance
    _flying(p, yaw=math.radians(177.0))
    assert p.at_target() is True


def test_at_target_yaw_error_wraps_across_the_antimeridian():
    # A world yaw of -179 goes on the wire as PX4 +179. Reported as PX4 -179, the raw difference
    # is 358 deg; wrapped it's -2, which is inside tolerance. Without the wrap this reads as
    # nearly a full turn away and the vehicle never "arrives."
    p = _flying(_pilot(), yaw=0.0)
    p.goto((0.0, 0.0, 5.0), yaw=math.radians(-179.0))
    assert math.degrees(_last_goto(p)[3]) == pytest.approx(179.0, abs=1e-9)
    _flying(p, yaw=math.radians(-179.0))
    assert p.at_target() is True


def test_benchmark_profile_is_unchanged_by_the_rewrite():
    """The CI benchmark matrix flies *the* one 4-waypoint mission, and its baselines are only
    comparable if the rewritten world-axes waypoints resolve to the same targets the old
    ``"N,E,alt;…"`` string did. This is what protects #39's numbers from this change.
    """
    old_mission = "20,0,5;20,20,8;0,20,5;0,0,5"  # N,E,alt [m], as the old flight driver parsed it
    new_mission = ((20.0, 0.0, 5.0), (20.0, -20.0, 8.0), (0.0, -20.0, 5.0), (0.0, 0.0, 5.0))  # world axes
    home_amsl = AMSL0 - REL_ALT0

    p = _flying(_pilot())
    for chunk, world in zip(old_mission.split(";"), new_mission, strict=True):
        north, east, wp_alt = (float(v) for v in chunk.split(","))
        # The old conversion, lifted verbatim from the deleted flight driver's waypoint phase.
        want_lat = LAT0 + math.degrees(north / 6378137.0)
        want_lon = LON0 + math.degrees(east / (6378137.0 * math.cos(math.radians(LAT0))))
        p.goto(world)
        lat, lon, alt_amsl, _ = _last_goto(p)
        assert lat == pytest.approx(want_lat, abs=1e-6)
        assert lon == pytest.approx(want_lon, abs=1e-6)
        assert alt_amsl == pytest.approx(home_amsl + wp_alt)


# ---- the mission: MISSION_ITEM_INT upload + Mission mode, issue #66 ------------------------


class _FakeMsg:
    """A received MAVLink message: a type, plus whatever fields the handler reads off it."""

    def __init__(self, type_, **fields):
        self._type = type_
        self.__dict__.update(fields)

    def get_type(self):
        return self._type


def _item(seq, command, lat=0.0, lon=0.0, alt=40.0):
    return MissionItem(
        seq=seq,
        frame=3,  # MAV_FRAME_GLOBAL_RELATIVE_ALT
        command=command,
        autocontinue=True,
        params=(0.0, 0.0, 0.0, math.nan),
        x=round(lat * 1e7),
        y=round(lon * 1e7),
        z=alt,
    )


def _box():
    """The shipped box mission's shape: NAV_TAKEOFF, four NAV_WAYPOINTs, NAV_RETURN_TO_LAUNCH."""
    return (
        _item(0, 22, 47.5566, -122.1643),
        _item(1, 16, 47.5575009, -122.1643),
        _item(2, 16, 47.5575009, -122.1629651),
        _item(3, 16, 47.5566, -122.1629651),
        _item(4, 16, 47.5566, -122.1643),
        _item(5, 20, 0.0, 0.0, alt=0.0),
    )


def _counts(p):
    return [a for c, a in p._mav.mav.calls if c == "mission_count"]


def _items_sent(p):
    return [a for c, a in p._mav.mav.calls if c == "mission_item_int"]


def test_upload_mission_only_lodges_until_the_pump_runs():
    p = _pilot()
    p.upload_mission(_box())
    assert not p._mav.mav.calls  # the verb lodges intent, as every other one does
    _service(p, now=100.0)
    assert len(_counts(p)) == 1
    assert _counts(p)[0][2] == 6  # p3 = item count


def test_mission_count_is_resent_until_acked():
    # A lost MISSION_COUNT stalls the whole handshake: PX4 drives everything after it, so there
    # is nothing else to retry.
    p = _pilot()
    p.upload_mission(_box())
    st = _service(p, now=100.0)
    _service(p, now=100.5, state=st)
    assert len(_counts(p)) == 1  # under a second: not yet
    _service(p, now=101.5, state=st)
    assert len(_counts(p)) == 2
    p._on_msg(_FakeMsg("MISSION_ACK", type=0))
    _service(p, now=110.0, state=st)
    assert len(_counts(p)) == 2  # acked: the retry stops


def test_mission_request_int_returns_that_item():
    p = _pilot()
    p.upload_mission(_box())
    p._on_msg(_FakeMsg("MISSION_REQUEST_INT", seq=2))
    (args,) = _items_sent(p)
    # (sysid, compid, seq, frame, command, current, autocontinue, p1..p4, x, y, z, mission_type)
    assert args[2] == 2
    assert args[3] == 3  # frame: GLOBAL_RELATIVE_ALT
    assert args[4] == 16  # NAV_WAYPOINT
    assert args[5] == 0  # current: 0 for every item
    assert args[11] == 475575009  # x = lat * 1e7, the integer field
    assert args[12] == -1221629651  # y = lon * 1e7
    assert args[13] == 40.0


def test_plain_mission_request_is_answered_too():
    p = _pilot()
    p.upload_mission(_box())
    p._on_msg(_FakeMsg("MISSION_REQUEST", seq=0))
    assert _items_sent(p)[0][4] == 22  # NAV_TAKEOFF


def test_request_outside_the_mission_is_ignored():
    p = _pilot()
    p.upload_mission(_box())
    p._on_msg(_FakeMsg("MISSION_REQUEST_INT", seq=99))
    assert not _items_sent(p)


def test_mission_ack_accepted_marks_uploaded():
    p = _pilot()
    p.upload_mission(_box())
    assert p.mission_uploaded() is False
    p._on_msg(_FakeMsg("MISSION_ACK", type=0))
    assert p.mission_uploaded() is True
    assert p.mission_ack() == 0


def test_rejected_mission_is_not_uploaded_but_stops_the_retry():
    # An error ack must settle the upload, or the pump loops forever on a mission PX4 refused.
    p = _pilot()
    p.upload_mission(_box())
    st = _service(p, now=100.0)
    p._on_msg(_FakeMsg("MISSION_ACK", type=13))  # MAV_MISSION_INVALID_SEQUENCE
    _service(p, now=110.0, state=st)
    assert p.mission_uploaded() is False
    assert p.mission_ack() == 13
    assert len(_counts(p)) == 1


def test_mission_reached_never_decreases():
    p = _pilot()
    p.upload_mission(_box())
    assert p.mission_reached() == -1
    p._on_msg(_FakeMsg("MISSION_ITEM_REACHED", seq=3))
    p._on_msg(_FakeMsg("MISSION_ITEM_REACHED", seq=1))  # a stale/duplicate report
    assert p.mission_reached() == 3


def test_mission_current_is_cached():
    p = _pilot()
    p.upload_mission(_box())
    p._on_msg(_FakeMsg("MISSION_CURRENT", seq=4))
    assert p.mission_current() == 4


def test_mission_complete_fires_on_the_last_waypoint_not_the_rtl():
    # The mission ends NAV_RETURN_TO_LAUNCH, seq 5, which hands over to Return To Launch (RTL) rather
    # than reporting an arrival: waiting on the final *item* would never return.
    p = _pilot()
    p.upload_mission(_box())
    assert p.mission_complete() is False
    p._on_msg(_FakeMsg("MISSION_ITEM_REACHED", seq=3))
    assert p.mission_complete() is False
    p._on_msg(_FakeMsg("MISSION_ITEM_REACHED", seq=4))  # the last NAV_WAYPOINT
    assert p.mission_complete() is True


def test_mission_complete_is_false_without_a_mission():
    assert _pilot().mission_complete() is False


def test_reupload_resets_the_progress_counters():
    p = _pilot()
    p.upload_mission(_box())
    p._on_msg(_FakeMsg("MISSION_ACK", type=0))
    p._on_msg(_FakeMsg("MISSION_ITEM_REACHED", seq=4))
    p.upload_mission(_box())
    assert p.mission_reached() == -1
    assert p.mission_current() == 0
    assert p.mission_uploaded() is False  # the new mission isn't acked yet


def test_upload_mission_reads_a_path(tmp_path):
    plan = tmp_path / "one.plan"
    plan.write_text(
        json.dumps(
            {
                "fileType": "Plan",
                "version": 1,
                "mission": {
                    "version": 2,
                    "plannedHomePosition": [47.5566, -122.1643, 0],
                    "items": [
                        {
                            "type": "SimpleItem",
                            "command": 16,
                            "frame": 3,
                            "autoContinue": True,
                            "doJumpId": 1,
                            "params": [0, 0, 0, None, 47.5575009, -122.1643, 40.0],
                        }
                    ],
                },
            }
        )
    )
    p = _pilot()
    p.upload_mission(plan)
    assert p.mission_count() == 1
    _service(p, now=100.0)
    assert _counts(p)[0][2] == 1


def test_start_mission_sets_the_mode_before_arming():
    # Same load-bearing ordering as takeoff(): PX4 confirms DO_SET_MODE *before* the arm command.
    from pymavlink import mavutil

    arm_cmd = mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
    p = _pilot()
    p._rel_alt = 0.0  # PX4 is publishing a position estimate
    p.upload_mission(_box())
    p.start_mission()
    assert p._want_mode == "Mission"
    t0 = p._arm_since
    st = _service(p, now=t0 + 10.0, state=None)
    assert not _sent(p, arm_cmd)  # settled, but the mode is still unconfirmed
    p._mode = "Mission"
    _service(p, now=t0 + 20.0, state=st)
    assert len(_sent(p, arm_cmd)) == 1
