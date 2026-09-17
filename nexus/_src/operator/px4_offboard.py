"""Px4Offboard: a thin, non-blocking pymavlink operator, the PX4 implementation of the Operator
role. A background daemon thread keeps the MAVLink link alive: it sends the Ground Control
Station (GCS) heartbeat plus the current MANUAL_CONTROL setpoint, services the outstanding mode and
arm requests, and caches telemetry. Every public verb is a **request**: it returns immediately and
the pump commands PX4 until telemetry confirms it, so no verb ever blocks the thread that drives the
sim. A script waits on the result instead, ``sim.wait_until(op.at_target, …)``. The arm and mode
recipe is the proven headless one: switch mode before arming, and wait until armable before
hammering arm.

The name refers to the **link**: PX4's offboard and onboard API on User Datagram Protocol (UDP)
port ``:14540``, where MAVSDK connects with ``-m onboard``, *not* PX4's OFFBOARD flight mode. It's
a separate link from the controller's Hardware In The Loop (HIL) lockstep on ``:4560``, mirroring
the real topology, the PX4 link map.
"""

from __future__ import annotations

import math
import os
import struct
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass

from nexus._src.core.schema import Setpoint

from .operator import BaseOperator
from .qgc_plan import NAV_WAYPOINT, MissionItem, Plan, read_plan

# Pin the MAVLink dialect before importing mavutil; this matches Px4MavlinkController. Because the
# package might eagerly import Px4Offboard, this can be the *first* pymavlink import in the process;
# the HIL controller's HIL_GPS uses the common-dialect `id`/`yaw` fields, so this file must set the
# dialect too or that later import is a cached no-op and HIL_GPS loses those fields.
os.environ.setdefault("MAVLINK20", "1")
os.environ.setdefault("MAVLINK_DIALECT", "common")

from pymavlink import mavutil

# PX4 px4_custom_mode main modes.
_PX4_MAIN = {"MANUAL": 1, "ALTCTL": 2, "POSCTL": 3, "AUTO": 4, "ACRO": 5, "OFFBOARD": 6, "STABILIZED": 7}
# Sub-modes of the auto main mode, Return To Launch (RTL) among them.
_AUTO = {"READY": 1, "TAKEOFF": 2, "LOITER": 3, "MISSION": 4, "RTL": 5, "LAND": 6}
# Friendly name -> (main mode name, sub mode int; 0 = no sub).
_MODES: dict[str, tuple[str, int]] = {
    "Manual": ("MANUAL", 0),
    "Altitude": ("ALTCTL", 0),
    "Position": ("POSCTL", 0),
    "Stabilized": ("STABILIZED", 0),
    "Offboard": ("OFFBOARD", 0),
    "Hold": ("AUTO", _AUTO["LOITER"]),
    "Takeoff": ("AUTO", _AUTO["TAKEOFF"]),
    "Mission": ("AUTO", _AUTO["MISSION"]),
    "RTL": ("AUTO", _AUTO["RTL"]),
    "Land": ("AUTO", _AUTO["LAND"]),
}
# Manual flight modes: PX4 rejects entering these unless a manual-control source, a joystick, is live,
# so set_mode auto-starts a neutral MANUAL_CONTROL stream first; see set_mode.
_MANUAL_MODES = {"Manual", "Altitude", "Position", "Stabilized", "Acro"}
_LANDED = {
    mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND: "ON_GROUND",
    mavutil.mavlink.MAV_LANDED_STATE_IN_AIR: "IN_AIR",
    mavutil.mavlink.MAV_LANDED_STATE_TAKEOFF: "TAKING_OFF",
    mavutil.mavlink.MAV_LANDED_STATE_LANDING: "LANDING",
}


@dataclass
class _RequestState:
    """The pump's private send-timing for the outstanding mode and arm requests: when each command was
    last put on the wire, and which phase a stick-gesture arm is in. Lives on the pump's stack, not
    on the operator, because nothing outside the pump has any business reading it.
    """

    last_mode: float = 0.0
    last_arm: float = 0.0
    last_count: float = 0.0  # when MISSION_COUNT last went out, the mission upload handshake
    gesture_since: float = 0.0  # when the current stick phase started; 0.0 = nothing held yet
    gesture_held: bool = False  # True = streaming the gesture, False = streaming neutral


# Spherical-earth radius for the local tangent-plane conversion [m].
_R_EARTH = 6378137.0

# The stick-gesture arm; see arm(gesture=True). PX4 arms on the *rising* edge of the gesture
# hysteresis, so a gesture it can't act on burns the one edge it gets and never re-fires until the
# sticks leave the gesture and come back. The pump so cycles hold -> neutral -> hold until the
# vehicle arms. Both windows are wall-clock, the same as everything else in the pump, while PX4
# measures COM_RC_ARM_HYST, 1 s, on its sim clock: generous in this operator's favour at any
# Real-Time Factor (RTF) over 1.
_GESTURE_HOLD_S = 3.0  # hold the gesture this long before dropping the edge
_GESTURE_GAP_S = 0.5  # then neutral this long, so PX4 registers the falling edge before the next rising one
_NEUTRAL_RC = {"roll": 0.0, "pitch": 0.0, "throttle": 0.5, "yaw": 0.0}
_GESTURE_RC = {"roll": 0.0, "pitch": 0.0, "throttle": 0.0, "yaw": 1.0}  # throttle down + yaw full right


def _wrap_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi), so a heading error never reads as most of a turn."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class _ClimbTarget:
    """What a :meth:`Px4Offboard.takeoff` is heading for: an altitude over the launch point [m]."""

    rel_alt: float


@dataclass
class _GotoTarget:
    """What a :meth:`Px4Offboard.goto` is heading for, in the frame PX4 reports: the commanded fix,
    the altitude over the launch point, and the commanded heading, where ``None`` means the caller
    commanded no yaw. ``yaw`` is the PX4-frame value that went on the wire, not the caller's world
    yaw, so arrival checks compare PX4 frame with PX4 frame against ``ATTITUDE``.
    """

    lat: float
    lon: float
    rel_alt: float
    yaw: float | None


class Px4Offboard(BaseOperator):
    def __init__(
        self,
        conn: str = "udpin:0.0.0.0:14540",
        *,
        arrive_m: float = 2.0,
        yaw_tol_rad: float = math.radians(5.0),
        alt_tol_m: float = 1.0,
        wait_clear_s: float = 5.0,
    ):
        """Configure the operator link; nothing opens a connection until entered.

        Args:
            conn: pymavlink connection string for the PX4 operator link. The default
                listens on UDP port 14540, the PX4 offboard and GCS API port.
            arrive_m: 3D arrival radius for :meth:`at_target` after a :meth:`goto`, in meters.
            yaw_tol_rad: Heading tolerance for :meth:`at_target` when the caller commanded a yaw, in radians.
            alt_tol_m: Altitude tolerance for :meth:`at_target` after a :meth:`takeoff`, in meters.
            wait_clear_s: Seconds the vehicle must be failure-free, and the initial settle
                period, before the pump sends an arm command.
        """
        super().__init__()  # BaseOperator, for _as_position_goal; nothing uses the mission and logging members
        self._conn_str = conn
        self._arrive_m = arrive_m
        self._yaw_tol_rad = yaw_tol_rad
        self._alt_tol_m = alt_tol_m
        self._wait_clear_s = wait_clear_s
        self._mav = None
        self._lock = threading.Lock()  # guards every send on the shared connection
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sysid = None
        self._compid = None
        # cached telemetry: the pump thread writes it, callers read it
        self._armed = False
        self._mode = ""
        self._landed = "UNKNOWN"
        self._rel_alt: float | None = None
        self._lat: float | None = None  # deg, from GLOBAL_POSITION_INT
        self._lon: float | None = None  # deg
        self._amsl: float | None = None  # m over mean sea level, the frame DO_REPOSITION resolves in
        self._yaw: float | None = None  # rad, world heading from ATTITUDE
        self._last_fail = 0.0
        self._rc: tuple[int, int, int, int] | None = None  # streamed MANUAL_CONTROL setpoint
        # Outstanding requests, serviced by the pump; see _service_requests. Each clears the
        # moment telemetry confirms it, so the pump never re-commands a state PX4 has already
        # left of its own accord: Takeoff mode hands over to Hold on arrival, and PX4 disarms
        # itself after landing; a latched request would fight the first and re-arm after the second.
        self._want_mode: str | None = None
        self._want_arm = False
        self._arm_gesture = False  # the outstanding arm is a stick gesture, not a command
        self._arm_since = 0.0  # when the caller requested the arm; the settle window starts here
        # The local tangent-plane anchor, lat0 [deg], lon0 [deg] and home altitude over mean sea level
        # [m], captured lazily on the first goto, and the target the last verb commanded, which
        # at_target measures against.
        self._anchor: tuple[float, float, float] | None = None
        self._target: _ClimbTarget | _GotoTarget | None = None
        # The uploaded mission; see upload_mission. The upload is a *handshake*, not a send: this
        # operator puts MISSION_COUNT on the wire and PX4 then asks for each item by seq, so the pump
        # drives it from _on_msg and the caller waits on mission_uploaded().
        self._mission_items: tuple[MissionItem, ...] = ()
        self._want_upload = False  # outstanding upload: keep re-sending MISSION_COUNT until acked
        self._mission_ack: int | None = None  # MISSION_ACK type; 0 = accepted, None = not settled
        self._reached = -1  # highest MISSION_ITEM_REACHED seq; -1 = none reached yet
        self._mission_current = 0  # most recent MISSION_CURRENT seq

    # ---- lifecycle ----
    def open(self) -> Px4Offboard:
        """Open the MAVLink link and start the pump, returning at once, with no wait for PX4.

        The non-blocking half of :meth:`__enter__`, for a caller that owns the waiting. ``Sim``
        uses it because PX4's clock is the sim's under lockstep: the sim has to keep stepping or
        no heartbeat ever arrives, so the wait has to be a stepping loop rather than a sleep.
        Poll :attr:`connected` to know when PX4 has answered.

        Returns:
            ``self``, so it chains the same way as ``__enter__``.
        """
        self._stop.clear()
        self._mav = mavutil.mavlink_connection(self._conn_str, source_system=255, source_component=240)
        self._thread = threading.Thread(target=self._pump, name="px4-offboard", daemon=True)
        self._thread.start()
        return self

    @property
    def connected(self) -> bool:
        """Whether PX4 has answered on the operator link: a heartbeat has arrived and the pump has
        learned the peer's system id, so every verb has somewhere to send.

        Returns:
            ``True`` once the first PX4 heartbeat has arrived on ``:14540``.
        """
        return self._sysid is not None

    def __enter__(self) -> Px4Offboard:
        self.open()
        self._await_heartbeat(timeout=30.0)
        return self

    def __exit__(self, *exc) -> None:
        """Stop the pump and close the link on exit; see ``close``."""
        self.close()

    def close(self) -> None:
        """Stop the background pump thread and close the MAVLink connection.

        Idempotent and safe to call after a partial start: signals the pump to stop,
        joins it, for up to 5 s, then closes the connection if ``open`` made one.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._mav is not None:
            self._mav.close()

    # ---- background pump: heartbeat + RC streaming + request service + telemetry cache ----
    def _pump(self) -> None:
        last_hb = last_rc = 0.0
        state = _RequestState()
        while not self._stop.is_set():
            now = time.monotonic()
            if now - last_hb > 1.0:
                with self._lock:
                    self._mav.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
                    )
                last_hb = now
            rc = self._rc
            if rc is not None and self._sysid is not None and now - last_rc > 0.05:  # ~20 Hz MANUAL_CONTROL
                with self._lock:
                    self._mav.mav.manual_control_send(self._sysid, rc[0], rc[1], rc[2], rc[3], 0)
                last_rc = now
            self._service_requests(now, state)
            msg = self._mav.recv_match(blocking=True, timeout=0.1)
            if msg is not None:
                self._on_msg(msg)

    def _service_requests(self, now: float, state: _RequestState) -> None:
        """Send the outstanding mode/arm commands, retrying until telemetry confirms them.

        The retry is what the blocking ``set_mode`` and ``arm`` used to do inline: under load PX4 can
        silently ignore a single command, because the command races the autopilot's state machine,
        or the vehicle isn't yet armable, and the vehicle simply stays put. Verification-based, so
        it's robust to the sim's real-time factor. Wall-clock pacing is right here because the
        operator is a remote GCS, not a node in the sim loop.

        Args:
            now: ``time.monotonic()`` for this pump iteration.
            state: The pump's private send-timing state.
        """
        if self._sysid is None:
            return
        # The mission upload starts with MISSION_COUNT and PX4 drives the rest, so a lost
        # MISSION_COUNT stalls the whole handshake with nothing to retry it, hence the same
        # once-a-second resend the mode and arm requests use, until MISSION_ACK settles it.
        if self._want_upload and now - state.last_count > 1.0:
            with self._lock:
                self._mav.mav.mission_count_send(
                    self._sysid, self._compid, len(self._mission_items), mavutil.mavlink.MAV_MISSION_TYPE_MISSION
                )
            state.last_count = now
        if self._want_mode is not None:
            if self._mode == self._want_mode:
                self._want_mode = None  # confirmed: stop commanding it
            elif now - state.last_mode > 1.0:
                main_name, sub = _MODES[self._want_mode]
                self._cmd(
                    mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    _PX4_MAIN[main_name],
                    sub,
                )
                state.last_mode = now
        if not self._want_arm:
            return
        if self._armed:
            self._want_arm = False  # confirmed: never re-arm after PX4's own post-landing disarm
            if self._arm_gesture:
                self._arm_gesture = False
                self.set_rc(**_NEUTRAL_RC)  # never leave the vehicle armed at throttle-down + full-right yaw
                state.gesture_held, state.gesture_since = False, 0.0
            return
        # The mode change goes *first* and telemetry must confirm it before arming: PX4's arming
        # checks are mode-dependent, and in the default manual mode the missing joystick leaves the
        # vehicle non-armable. Armable also means PX4 is publishing a position estimate, so the
        # Extended Kalman Filter (EKF) has converged, and has been failure-free for wait_clear_s:
        # arming during the convergence transient trips the strict pre-arm health check, so the pump
        # doesn't hammer it. The request counts as "just failed" so the *first* try waits out a full
        # settle period.
        clear_for = now - max(self._last_fail, self._arm_since)
        armable = self._want_mode is None and self._rel_alt is not None and clear_for >= self._wait_clear_s
        if not armable:
            if self._arm_gesture and state.gesture_held:
                self.set_rc(**_NEUTRAL_RC)  # a gesture PX4 can't act on burns the one edge it gets
                state.gesture_held, state.gesture_since = False, 0.0
            return
        if self._arm_gesture:
            # Cycle the sticks so PX4 always has a fresh rising edge to arm on; see _GESTURE_HOLD_S.
            phase_s = now - state.gesture_since
            if not state.gesture_held and (state.gesture_since == 0.0 or phase_s >= _GESTURE_GAP_S):
                self.set_rc(**_GESTURE_RC)  # the rising edge PX4 arms on
                state.gesture_held, state.gesture_since = True, now
            elif state.gesture_held and phase_s >= _GESTURE_HOLD_S:
                self.set_rc(**_NEUTRAL_RC)  # denied: drop the edge so the next hold re-creates it
                state.gesture_held, state.gesture_since = False, now
            return
        if now - state.last_arm > 1.5:
            self._cmd(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1.0)
            state.last_arm = now

    def _on_msg(self, msg) -> None:
        t = msg.get_type()
        if t == "HEARTBEAT" and msg.get_srcSystem() == 1:
            if self._sysid is None:
                self._sysid, self._compid = self._mav.target_system, self._mav.target_component
            self._armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            self._mode = self._decode_mode(msg.custom_mode)
        elif t == "GLOBAL_POSITION_INT":
            self._rel_alt = msg.relative_alt / 1000.0
            self._lat = msg.lat / 1e7
            self._lon = msg.lon / 1e7
            self._amsl = msg.alt / 1000.0
        elif t == "ATTITUDE":
            self._yaw = msg.yaw
        elif t == "EXTENDED_SYS_STATE":
            self._landed = _LANDED.get(msg.landed_state, "UNKNOWN")
        elif t == "STATUSTEXT":
            txt = msg.text if isinstance(msg.text, str) else msg.text.decode(errors="ignore")
            if any(k in txt for k in ("Fail", "denied", "Denied", "reject", "interference")):
                self._last_fail = time.monotonic()
        elif t in ("MISSION_REQUEST_INT", "MISSION_REQUEST"):
            # PX4 pulls the mission item by item. It asks with the MISSION_REQUEST_INT form; the plain
            # form costs one name in this tuple and makes this operator robust to a peer that doesn't.
            self._send_mission_item(msg.seq)
        elif t == "MISSION_ACK":
            # Settles the upload either way: on an error ack, stop resending MISSION_COUNT rather
            # than loop forever on a mission PX4 has already refused.
            self._mission_ack = msg.type
            self._want_upload = False
        elif t == "MISSION_ITEM_REACHED":
            self._reached = max(self._reached, msg.seq)
        elif t == "MISSION_CURRENT":
            self._mission_current = msg.seq

    def _send_mission_item(self, seq: int) -> None:
        """Answer PX4's request for one mission item; a no-op if it asks outside the mission."""
        if not 0 <= seq < len(self._mission_items):
            return
        it = self._mission_items[seq]
        with self._lock:
            self._mav.mav.mission_item_int_send(
                self._sysid,
                self._compid,
                it.seq,
                it.frame,
                it.command,
                0,  # current: 0 for every item; PX4 runs the mission from the start
                int(it.autocontinue),
                it.params[0],
                it.params[1],
                it.params[2],
                it.params[3],
                it.x,
                it.y,
                it.z,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
            )

    @staticmethod
    def _decode_mode(custom_mode: int) -> str:
        main = (custom_mode >> 16) & 0xFF
        sub = (custom_mode >> 24) & 0xFF
        for name, (m, s) in _MODES.items():
            if _PX4_MAIN[m] == main and (s == 0 or s == sub):
                return name
        return f"main={main},sub={sub}"

    def _await_heartbeat(self, timeout: float) -> None:
        """Sleep-poll until PX4 answers, for a caller with nothing else to drive.

        A sim-driving caller must *not* use this: under lockstep the sim owns PX4's clock, so a
        caller that sleeps here stops the sim and PX4 never sends the heartbeat it's waiting for.
        ``Sim.operator`` steps the sim instead, GH #70.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.connected:
                return
            time.sleep(0.1)
        raise TimeoutError("no PX4 heartbeat on the operator link")

    def _cmd(self, command, *params) -> None:
        with self._lock:
            self._mav.mav.command_long_send(
                self._sysid, self._compid, command, 0, *(list(params) + [0.0] * (7 - len(params)))
            )

    def _cmd_int(self, command, *params) -> None:
        """Send a COMMAND_INT in ``MAV_FRAME_GLOBAL``, the form DO_REPOSITION needs, because its
        lat and lon ride the integer x and y fields; a float param would lose the last metres of precision.
        """
        with self._lock:
            self._mav.mav.command_int_send(
                self._sysid, self._compid, mavutil.mavlink.MAV_FRAME_GLOBAL, command, 0, 0, *params
            )

    # ---- the local tangent plane: world axes, Newton Forward-Left-Up (FLU) and Z-up -> WGS84 ----
    def _anchor_now(self) -> tuple[float, float, float]:
        """The tangent-plane anchor, captured on first use from the cached position.

        The anchor is whatever PX4 was reporting the first time a ``goto`` needed one. A script
        waits for the climb before its first ``goto``, so in practice that's "after takeoff
        settles"; a script that never takes off anchors at its current position instead.

        Returns:
            ``(lat0 [deg], lon0 [deg], home_amsl [m])``, where ``home_amsl`` is the launch point's
            altitude over mean sea level, the frame DO_REPOSITION altitudes resolve against.

        Raises:
            RuntimeError: No position has arrived on the operator link yet.
        """
        if self._anchor is None:
            if self._lat is None or self._lon is None or self._amsl is None or self._rel_alt is None:
                raise RuntimeError("no PX4 position yet: goto needs a GLOBAL_POSITION_INT to anchor against")
            self._anchor = (self._lat, self._lon, self._amsl - self._rel_alt)
        return self._anchor

    def _to_global(self, x: float, y: float) -> tuple[float, float]:
        """Convert a world-axes offset from the anchor to a WGS84 fix on a spherical-earth local
        tangent plane. Newton's world frame is FLU with north along +x, which forces east onto −y.

        Args:
            x: Offset along the world +x axis, north, from the anchor, in meters.
            y: Offset along the world +y axis, west, from the anchor, in meters.

        Returns:
            ``(lat, lon)`` in degrees.
        """
        lat0, lon0, _ = self._anchor_now()
        north, east = x, -y
        lat = lat0 + math.degrees(north / _R_EARTH)
        lon = lon0 + math.degrees(east / (_R_EARTH * math.cos(math.radians(lat0))))
        return lat, lon

    # ---- operator verbs: every one returns immediately; the caller waits on telemetry ----
    def set_mode(self, mode: str) -> None:
        """Request a flight mode. Returns at once; the pump commands it until telemetry confirms.

        Entering a manual mode requires a live manual-control source, so a neutral MANUAL_CONTROL
        stream is auto-started first. Observe the result with :meth:`mode`.

        Args:
            mode: Friendly mode name; one of the keys of the mode table, for example ``"Manual"``,
                ``"Altitude"``, ``"Position"``, ``"Offboard"``, ``"Hold"``, ``"Takeoff"``,
                ``"Mission"``, ``"RTL"`` or ``"Land"``.

        Raises:
            ValueError: If ``mode`` isn't a known mode name.
        """
        if mode not in _MODES:
            raise ValueError(f"unknown mode {mode!r}; known: {sorted(_MODES)}")
        # PX4 rejects a switch *into* a manual mode unless a manual-control source is live, so
        # ensure the neutral MANUAL_CONTROL stream is running; the pump streams self._rc.
        if mode in _MANUAL_MODES and self._rc is None:
            self.set_rc(roll=0.0, pitch=0.0, throttle=0.5, yaw=0.0)
        self._want_mode = mode

    def arm(self, *, gesture: bool = False) -> None:
        """Request arming. Returns at once; the pump arms once the vehicle is armable.

        Armable means telemetry has confirmed any pending mode change, PX4 is publishing a position
        estimate, so the EKF has converged, and it has been failure-free for the configured settle
        period. Observe the result with :meth:`is_armed`.

        Args:
            gesture: Arm with the pilot's throttle-down plus yaw-right **stick gesture** rather than
                ``MAV_CMD_COMPONENT_ARM_DISARM``: the manual path a real pilot flies. The pump
                holds neutral sticks until the vehicle is armable, then presents the gesture, so
                PX4 only ever gets an edge it can act on. Needs a live manual-control source, which
                this starts if none is running. The default, ``False``, is the command arm, which is
                what :meth:`takeoff` and every autonomous script use.
        """
        self._arm_since = time.monotonic()  # the settle window starts now
        self._want_arm = True
        self._arm_gesture = gesture
        if gesture and self._rc is None:
            self.set_rc(**_NEUTRAL_RC)  # PX4 needs a live manual-control source before a gesture can arm

    def disarm(self) -> bool:
        """Send a single disarm command, fire-and-forget and not verified.

        Returns:
            ``True`` once ``MAV_CMD_COMPONENT_ARM_DISARM`` has gone out.
        """
        self._want_arm = False  # drop any outstanding arm request, or the pump would undo this
        self._arm_gesture = False  # … and a dropped request must not leave the pump gesturing
        self._cmd(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0.0)
        return True

    def set_rc(self, *, roll: float = 0.0, pitch: float = 0.0, throttle: float = 0.5, yaw: float = 0.0) -> None:
        """Set the streamed MANUAL_CONTROL setpoint. roll/pitch/yaw in [-1,1]; throttle in [0,1].

        Stores the setpoint; the background pump streams it at ~20 Hz until changed or
        cleared with ``stop_rc``. This clamps the values to range, then scales them to the
        MANUAL_CONTROL integer fields.

        Args:
            roll: Roll stick, clamped to [-1, 1].
            pitch: Pitch stick, clamped to [-1, 1].
            throttle: Throttle stick, clamped to [0, 1]; 0.5 is mid-stick.
            yaw: Yaw stick, clamped to [-1, 1].
        """

        def clamp(v, lo, hi):
            return max(lo, min(hi, v))

        x = int(clamp(pitch, -1.0, 1.0) * 1000)
        y = int(clamp(roll, -1.0, 1.0) * 1000)
        z = int(clamp(throttle, 0.0, 1.0) * 1000)
        r = int(clamp(yaw, -1.0, 1.0) * 1000)
        self._rc = (x, y, z, r)

    def stop_rc(self) -> None:
        """Stop streaming the MANUAL_CONTROL setpoint; the pump goes quiet until ``set_rc``."""
        self._rc = None

    def param_set(self, name: str, value: float) -> None:
        """Set a REAL32 PX4 parameter: a fire-and-forget ``PARAM_SET`` that awaits no acknowledgement.

        Float-only on purpose: an integer parameter doesn't ride the wire as a float, so it gets
        its own verb rather than a type argument this one can't honour; see :meth:`param_set_int`.

        Args:
            name: Parameter name, for example ``"MIS_TAKEOFF_ALT"``.
            value: New value, sent as a float.
        """
        with self._lock:
            self._mav.mav.param_set_send(
                self._sysid, self._compid, name.encode(), float(value), mavutil.mavlink.MAV_PARAM_TYPE_REAL32
            )

    def param_set_int(self, name: str, value: int) -> None:
        """Set an INT32 PX4 parameter: a fire-and-forget ``PARAM_SET`` that awaits no acknowledgement.

        PARAM_SET carries every value in one float field, and PX4 reads an integer parameter out of
        that field's **bits** rather than its numeric value. So this reinterprets the int rather than
        casting it: sending ``float(1)`` would have PX4 read 1065353216.

        Args:
            name: Parameter name, for example ``"COM_RC_IN_MODE"``.
            value: New value, sent as the INT32 bit pattern of the float field.
        """
        bits = struct.unpack("<f", struct.pack("<i", int(value)))[0]
        with self._lock:
            self._mav.mav.param_set_send(
                self._sysid, self._compid, name.encode(), bits, mavutil.mavlink.MAV_PARAM_TYPE_INT32
            )

    def takeoff(self, alt_m: float) -> None:
        """Set the takeoff altitude, request Takeoff mode, then request arming. Returns at once.

        The ordering is load-bearing and this method owns it: telemetry confirms
        the ``DO_SET_MODE`` for Takeoff mode **before** the arm command, which is why an autonomous script never
        calls :meth:`arm` itself. PX4 climbs on arming; wait for the climb with :meth:`at_target`.

        Args:
            alt_m: Target takeoff altitude in meters, written to ``MIS_TAKEOFF_ALT``.
        """
        self.param_set("MIS_TAKEOFF_ALT", alt_m)
        self._target = _ClimbTarget(rel_alt=alt_m)
        self.set_mode("Takeoff")
        self.arm()

    def land(self) -> None:
        """Request Land mode to land the vehicle in place. Returns at once."""
        self._target = None  # nothing to arrive at; at_target goes False
        self.set_mode("Land")

    # ---- the mission: a QGroundControl .plan, uploaded and flown in Mission mode ----
    def upload_mission(self, plan: Plan | Sequence[MissionItem] | str | os.PathLike) -> None:
        """Upload a mission to PX4. Returns at once; wait for it with :meth:`mission_uploaded`.

        The items go on the wire **verbatim**, in the geodetic frame the ``.plan`` authored them in;
        nothing converts through the sim's world frame, so a plan flies where QGroundControl drew it.

        Upload is a handshake rather than a send: this lodges the items, the pump puts
        ``MISSION_COUNT`` on the wire, PX4 then asks for each item by ``seq``, and it ends with
        PX4's ``MISSION_ACK``. Re-uploading replaces the mission and resets the progress counters,
        so a second mission never reports the first one's arrivals.

        Args:
            plan: A :class:`~nexus._src.operator.qgc_plan.Plan`, a bare sequence of
                :class:`~nexus._src.operator.qgc_plan.MissionItem`, or a path to a ``.plan``
                file, read with :func:`~nexus._src.operator.qgc_plan.read_plan`.
        """
        if isinstance(plan, (str, os.PathLike)):
            plan = read_plan(plan)
        self._mission_items = tuple(plan.items if isinstance(plan, Plan) else plan)
        self._mission_ack = None
        self._reached = -1
        self._mission_current = 0
        self._want_upload = True

    def mission_uploaded(self) -> bool:
        """Report whether PX4 has accepted the uploaded mission.

        Returns:
            ``True`` once ``MISSION_ACK`` reported acceptance. Stays ``False`` while the handshake
            is in flight, and also when PX4 **rejected** the mission; check :meth:`mission_ack`
            to tell a "not yet" from a "refused" answer.
        """
        return self._mission_ack == mavutil.mavlink.MAV_MISSION_ACCEPTED

    def mission_ack(self) -> int | None:
        """Report PX4's ``MISSION_ACK`` result code.

        Returns:
            The MAV_MISSION_RESULT value, where ``0`` = accepted, or ``None`` if PX4 hasn't
            acknowledged an upload yet.
        """
        return self._mission_ack

    def mission_count(self) -> int:
        """Report how many items are in the uploaded mission.

        Returns:
            The item count, or ``0`` before any upload.
        """
        return len(self._mission_items)

    def mission_reached(self) -> int:
        """Report the highest mission item PX4 says it has reached.

        Returns:
            The sequence number from the most recent ``MISSION_ITEM_REACHED``, or ``-1`` if none has
            arrived yet. Never decreases within one mission.
        """
        return self._reached

    def mission_current(self) -> int:
        """Report the item PX4 is flying to.

        Returns:
            The sequence number from the most recent ``MISSION_CURRENT``, or ``0`` before the first.
        """
        return self._mission_current

    def mission_complete(self) -> bool:
        """Report whether the vehicle has reached the mission's last **waypoint**.

        Measured on the last ``NAV_WAYPOINT``, not the last item. A mission that ends in
        ``NAV_RETURN_TO_LAUNCH`` hands the flight to Return To Launch (RTL) instead of reporting an
        arrival there, so waiting on the final item could never return. Wait out the RTL tail with
        :meth:`landed_state` instead.

        Returns:
            ``True`` once the vehicle has reached the final waypoint; ``False`` with no uploaded
            mission, or one that carries no waypoint at all.
        """
        last_wp = [it.seq for it in self._mission_items if it.command == NAV_WAYPOINT]
        return bool(last_wp) and self._reached >= last_wp[-1]

    def start_mission(self) -> None:
        """Request Mission mode, then request arming. Returns at once.

        The ordering is load-bearing and this method owns it, exactly as :meth:`takeoff` does:
        telemetry confirms the ``DO_SET_MODE`` for Mission mode **before** the arm command, which is the proven
        headless recipe. PX4 flies the mission on arming, starting from its ``NAV_TAKEOFF`` item.

        Upload the mission first and wait for :meth:`mission_uploaded`: PX4 refuses Mission mode
        while it has no valid mission, so engaging early just burns retries.
        """
        self.set_mode("Mission")
        self.arm()

    def goto(self, setpoint: Setpoint, *, yaw: float | None = None) -> None:
        """Fly to a world-frame position. Returns at once; wait for it with :meth:`at_target`.

        Commanded as ``DO_REPOSITION``: the same guidance PX4 runs for a GCS "fly here" request, and
        what it latches until the next command, so this goes out once rather than streamed. The
        position is in **world axes**, Newton FLU and Z-up, the same tuples ``InProcessOperator``
        takes, resolved against an anchor captured on the first call; see :meth:`_anchor_now`.

        Yaw is in world axes too, and this reflects it on the way out for the same reason it
        reflects the position: a right-handed yaw about world +z turns +x toward +y, which is north
        toward **west**, while PX4's North-East-Down (NED) heading is positive north toward
        **east**. Leaving the position in one frame and the heading in the other is the trap #61 was.

        Args:
            setpoint: The target: a :class:`~nexus._src.core.schema.PositionGoal` or a bare
                ``(x, y, z)`` world-frame position in meters, with ``z`` over the launch point.
            yaw: Optional world heading in radians, right-handed about +z, so +90° faces +y =
                west; overrides a ``PositionGoal``'s own ``yaw``. ``None`` leaves the heading
                to PX4.

        Raises:
            RuntimeError: No position has arrived on the operator link yet.
            TypeError: ``setpoint`` isn't a ``PositionGoal`` or an ``(x, y, z)`` position.
        """
        goal = self._as_position_goal(setpoint)
        x, y, z = goal.pos
        yaw_world = yaw if yaw is not None else goal.yaw
        yaw_px4 = None if yaw_world is None else _wrap_pi(-float(yaw_world))
        _, _, home_amsl = self._anchor_now()
        lat, lon = self._to_global(x, y)
        self._cmd_int(
            mavutil.mavlink.MAV_CMD_DO_REPOSITION,
            -1.0,  # p1 ground speed: -1 = default
            1.0,  # p2 bitmask: change to Hold on arrival
            0.0,  # p3 reserved
            float("nan") if yaw_px4 is None else yaw_px4,  # p4 yaw [rad], PX4 frame
            round(lat * 1e7),
            round(lon * 1e7),
            home_amsl + z,
        )
        # Store what went on the wire: ATTITUDE reports in PX4's frame, so at_target compares
        # PX4 against PX4 and never needs the conversion a second time.
        self._target = _GotoTarget(lat=lat, lon=lon, rel_alt=z, yaw=yaw_px4)

    def at_target(self) -> bool:
        """Report whether the vehicle has reached what the last verb commanded.

        Measured against **what PX4 reports**, the cached ``GLOBAL_POSITION_INT`` and ``ATTITUDE``,
        never the sim's ground truth. The operator is a remote GCS; reading sim state here would
        let it see a position the autopilot doesn't have.

        Returns:
            ``True`` once inside the arrival tolerances; ``False`` while still en route, with no
            target, because no verb commanded one or the last verb was :meth:`land`, or before the
            first position report.
        """
        target = self._target
        if target is None or self._rel_alt is None:
            return False
        if isinstance(target, _ClimbTarget):
            return abs(self._rel_alt - target.rel_alt) <= self._alt_tol_m
        if self._lat is None or self._lon is None:
            return False
        lat0, _, _ = self._anchor_now()
        m_per_deg = math.pi / 180.0 * _R_EARTH
        d_n = (self._lat - target.lat) * m_per_deg
        d_e = (self._lon - target.lon) * m_per_deg * math.cos(math.radians(lat0))
        d_z = self._rel_alt - target.rel_alt
        if math.sqrt(d_n * d_n + d_e * d_e + d_z * d_z) >= self._arrive_m:
            return False
        if target.yaw is None:
            return True
        if self._yaw is None:
            return False
        return abs(_wrap_pi(self._yaw - target.yaw)) < self._yaw_tol_rad

    # ---- telemetry: the arm, mode and landed state the HIL link can't see ----
    def is_armed(self) -> bool:
        """Report the cached arming state from the most recent heartbeat.

        Returns:
            ``True`` if the vehicle's last heartbeat reported the safety-armed flag.
        """
        return self._armed

    def mode(self) -> str:
        """Report the cached flight mode decoded from the most recent heartbeat.

        Returns:
            The friendly mode name, for example ``"Takeoff"``, or a ``"main=...,sub=..."``
            string if the mode isn't in the known table.
        """
        return self._mode

    def landed_state(self) -> str:
        """Report the cached landed state from EXTENDED_SYS_STATE.

        Returns:
            ``"ON_GROUND"``, ``"IN_AIR"``, ``"TAKING_OFF"``, ``"LANDING"``, or
            ``"UNKNOWN"`` if not yet reported.
        """
        return self._landed

    def relative_altitude(self) -> float | None:
        """Report the cached altitude over the launch point, in meters.

        Returns:
            Relative altitude in meters from GLOBAL_POSITION_INT, or ``None`` if no
            position has arrived yet.
        """
        return self._rel_alt
