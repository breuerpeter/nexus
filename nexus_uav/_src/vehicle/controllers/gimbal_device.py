"""The MAVLink end of a gimbal device: what a datagram on the gimbal's link means, and what the device sends.

:class:`GimbalDevice` speaks the MAVLink gimbal protocol, version 2, as a gimbal device, over the
User Datagram Protocol (UDP). It opens no socket and reads no clock: it takes a datagram and gives the setpoint and
the reply datagrams, and it builds the heartbeat and the status the gimbal's controller sends at their rates.
"""

from __future__ import annotations

import math
from typing import NamedTuple

PORT = 13280
"""The UDP port a gimbal device listens on, plus the run's instance."""


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
    """

    def __init__(self, *, roll=(math.nan, math.nan), pitch=(math.nan, math.nan), yaw=(math.nan, math.nan)):
        self.roll, self.pitch, self.yaw = tuple(roll), tuple(pitch), tuple(yaw)

    def handle(self, datagram: bytes) -> tuple[Setpoint | None, list[bytes]]:
        """Take one datagram off the link.

        Args:
            datagram: The bytes of one UDP datagram.

        Returns:
            The setpoint the datagram carries, ``None`` when it carries none, and the datagrams to answer with.
        """
        return None, []

    def heartbeat(self) -> bytes:
        """The device's ``HEARTBEAT``, as a datagram."""
        return b""

    def status(self, time_ms: int, attitude, rate) -> bytes:
        """The device's ``GIMBAL_DEVICE_ATTITUDE_STATUS``, as a datagram.

        Args:
            time_ms: The sim time [ms].
            attitude: The camera's attitude against North, East, Down, ``(x, y, z, w)``.
            rate: The camera's rate in its own frame [rad/s].
        """
        return b""


__all__ = ["PORT", "GimbalDevice", "Setpoint"]
