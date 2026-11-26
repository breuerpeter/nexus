"""The MAVLink end of a gimbal device: what a datagram on the gimbal's link means, and what the device sends.

:class:`GimbalDevice` speaks the MAVLink gimbal protocol, version 2, as a gimbal device, over the
User Datagram Protocol (UDP). It opens no socket and reads no clock: it takes a datagram and gives the setpoint and
the reply datagrams, and it builds the heartbeat and the status the gimbal's controller sends at their rates.

The device is component 154 of the system that first speaks to it, its gimbal manager's. It takes
``GIMBAL_DEVICE_SET_ATTITUDE`` and a ``COMMAND_LONG`` that requests its ``GIMBAL_DEVICE_INFORMATION``, and acts on
no other message. It holds roll and pitch against the horizon whatever the roll and pitch lock flags say, and
claims no retract and no neutral position, so it reads neither flag.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from pymavlink.dialects.v20 import common as mavlink

PORT = 13280
"""The UDP port a gimbal device listens on, plus the run's instance."""

COMPONENT = mavlink.MAV_COMP_ID_GIMBAL
"""The device's MAVLink component id, 154."""

_NAN = math.nan
# What every status reports beside the yaw lock: roll and pitch hold against the horizon, and the attitude it
# carries stands against North, a frame the device also takes a setpoint in.
_STATUS_FLAGS = (
    mavlink.GIMBAL_DEVICE_FLAGS_ROLL_LOCK
    | mavlink.GIMBAL_DEVICE_FLAGS_PITCH_LOCK
    | mavlink.GIMBAL_DEVICE_FLAGS_YAW_IN_EARTH_FRAME
    | mavlink.GIMBAL_DEVICE_FLAGS_ACCEPTS_YAW_IN_EARTH_FRAME
)
# What the gimbal can do: three axes, roll and pitch held against the horizon, and yaw against the vehicle's
# heading or against North, in either frame.
_CAPABILITIES = (
    mavlink.GIMBAL_DEVICE_CAP_FLAGS_HAS_ROLL_AXIS
    | mavlink.GIMBAL_DEVICE_CAP_FLAGS_HAS_ROLL_LOCK
    | mavlink.GIMBAL_DEVICE_CAP_FLAGS_HAS_PITCH_AXIS
    | mavlink.GIMBAL_DEVICE_CAP_FLAGS_HAS_PITCH_LOCK
    | mavlink.GIMBAL_DEVICE_CAP_FLAGS_HAS_YAW_AXIS
    | mavlink.GIMBAL_DEVICE_CAP_FLAGS_HAS_YAW_FOLLOW
    | mavlink.GIMBAL_DEVICE_CAP_FLAGS_HAS_YAW_LOCK
    | mavlink.GIMBAL_DEVICE_CAP_FLAGS_SUPPORTS_YAW_IN_EARTH_FRAME
)


class Setpoint(NamedTuple):
    """One setpoint off the link, as the gimbal's law takes it."""

    attitude: tuple[float, float, float, float]
    """The attitude that replaces the target, ``(x, y, z, w)``; NaN keeps the target."""
    rate: tuple[float, float, float]
    """The rate that moves the target [rad/s]."""
    locked: bool
    """The yaw lock bit: whether yaw holds against North, not the vehicle's heading."""
    earth: bool
    """Whether the attitude's heading is against North, not the vehicle's heading."""


class GimbalDevice:
    """A MAVLink gimbal device's side of its link.

    Args:
        roll: The lowest and the highest roll angle the gimbal reaches [rad], NaN for an axis with no limit.
        pitch: The same for pitch.
        yaw: The same for yaw.

    Attributes:
        system: The MAVLink system the device is a component of, its manager's: the system of the first message
            it took. ``None`` until a message arrives, and the device has nothing to send before.
        locked: The yaw lock bit in force, from the last setpoint, which the status reports.
    """

    def __init__(self, *, roll=(_NAN, _NAN), pitch=(_NAN, _NAN), yaw=(_NAN, _NAN)):
        self._limits = tuple(float(angle) for axis in (roll, pitch, yaw) for angle in axis)
        self._mav = mavlink.MAVLink(None, srcSystem=0, srcComponent=COMPONENT)
        self.system: int | None = None
        self.locked = False

    def handle(self, datagram: bytes, time_ms: int = 0) -> tuple[Setpoint | None, list[bytes]]:
        """Take one datagram off the link.

        Args:
            datagram: The bytes of one UDP datagram.
            time_ms: The sim time [ms], which the device's information carries.

        Returns:
            The setpoint the datagram carries, the last one when it carries more than one and ``None`` when it
            carries none, and the datagrams to answer with.
        """
        setpoint, replies = None, []
        # A parser per datagram: a datagram holds whole messages, so bytes that are no MAVLink end with it.
        parser = mavlink.MAVLink(None)
        parser.robust_parsing = True
        for message in parser.parse_buffer(datagram) or ():
            kind = message.get_type()
            if kind == "BAD_DATA":
                continue
            if self.system is None:
                self.system = self._mav.srcSystem = message.get_srcSystem()
            if kind == "GIMBAL_DEVICE_SET_ATTITUDE" and self._addressed(message):
                setpoint = self._setpoint(message)
            elif kind == "COMMAND_LONG" and message.target_system in (0, self.system):
                replies += self._answer(message, time_ms)
        return setpoint, replies

    def _addressed(self, message) -> bool:
        """Whether `message` names this device: its system or every system, and its component or every one."""
        return message.target_system in (0, self.system) and message.target_component in (0, COMPONENT)

    def _setpoint(self, message) -> Setpoint:
        """The setpoint of one ``GIMBAL_DEVICE_SET_ATTITUDE``. A frame flag names the frame of the attitude's
        heading, and with neither or both the lock flag names it.
        """
        w, x, y, z = (float(v) for v in message.q)
        attitude = (_NAN,) * 4 if any(math.isnan(v) for v in (x, y, z, w)) else (x, y, z, w)
        rates = (message.angular_velocity_x, message.angular_velocity_y, message.angular_velocity_z)
        rate = tuple(0.0 if math.isnan(v) else float(v) for v in rates)
        self.locked = bool(message.flags & mavlink.GIMBAL_DEVICE_FLAGS_YAW_LOCK)
        in_vehicle = bool(message.flags & mavlink.GIMBAL_DEVICE_FLAGS_YAW_IN_VEHICLE_FRAME)
        in_earth = bool(message.flags & mavlink.GIMBAL_DEVICE_FLAGS_YAW_IN_EARTH_FRAME)
        return Setpoint(attitude, rate, self.locked, in_earth if in_earth != in_vehicle else self.locked)

    def _answer(self, command, time_ms: int) -> list[bytes]:
        """The answer to one ``COMMAND_LONG``: the device's information behind an accepted ``COMMAND_ACK`` for a
        request for it, an unsupported ``COMMAND_ACK`` for any other command that names the device, and nothing
        for a command that names every component or another one.
        """
        if command.target_component not in (0, COMPONENT):
            return []
        requested = None
        # A NaN or infinite message id names no message.
        if command.command == mavlink.MAV_CMD_REQUEST_MESSAGE and math.isfinite(command.param1):
            requested = round(command.param1)
        if requested == mavlink.MAVLINK_MSG_ID_GIMBAL_DEVICE_INFORMATION:
            return [self._ack(command, mavlink.MAV_RESULT_ACCEPTED), self._information(time_ms)]
        if command.target_component == COMPONENT:
            return [self._ack(command, mavlink.MAV_RESULT_UNSUPPORTED)]
        return []

    def _ack(self, command, result: int) -> bytes:
        """A ``COMMAND_ACK`` of `command` with `result`, addressed to its sender."""
        ack = mavlink.MAVLink_command_ack_message(
            command.command, result, 0, 0, command.get_srcSystem(), command.get_srcComponent()
        )
        return ack.pack(self._mav)

    def _information(self, time_ms: int) -> bytes:
        """The device's ``GIMBAL_DEVICE_INFORMATION``: its limits and what it can do."""
        capabilities = _CAPABILITIES
        if math.isnan(self._limits[4]) and math.isnan(self._limits[5]):  # no yaw limit: it turns without end
            capabilities |= mavlink.GIMBAL_DEVICE_CAP_FLAGS_SUPPORTS_INFINITE_YAW
        information = mavlink.MAVLink_gimbal_device_information_message(
            time_ms, b"nexus", b"gimbal", b"", 0, 0, 0, capabilities, 0, *self._limits
        )
        return information.pack(self._mav)

    def heartbeat(self) -> bytes:
        """The device's ``HEARTBEAT``, as a datagram."""
        heartbeat = mavlink.MAVLink_heartbeat_message(
            mavlink.MAV_TYPE_GIMBAL, mavlink.MAV_AUTOPILOT_INVALID, 0, 0, mavlink.MAV_STATE_ACTIVE, 3
        )
        return heartbeat.pack(self._mav)

    def status(self, time_ms: int, attitude, rate) -> bytes:
        """The device's ``GIMBAL_DEVICE_ATTITUDE_STATUS``, as a datagram, for every component of every system.

        Args:
            time_ms: The sim time [ms].
            attitude: The camera's attitude against North, East, Down, ``(x, y, z, w)``.
            rate: The camera's rate in its own frame [rad/s].
        """
        x, y, z, w = (float(v) for v in attitude)
        flags = _STATUS_FLAGS | (mavlink.GIMBAL_DEVICE_FLAGS_YAW_LOCK if self.locked else 0)
        status = mavlink.MAVLink_gimbal_device_attitude_status_message(
            0, 0, time_ms, flags, [w, x, y, z], *(float(v) for v in rate), 0, _NAN, _NAN
        )
        return status.pack(self._mav)


__all__ = ["COMPONENT", "PORT", "GimbalDevice", "Setpoint"]
