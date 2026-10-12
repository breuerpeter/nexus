"""The neutral, typed vocabulary that flows between components.

A signal that lives on the device takes a Warp struct this module declares, or a Warp value type. The
hot-loop state is the physics backend's live ``newton.State``, which core never imports; the shared
``state.body_f`` device buffer realizes ``Wrench``, see the shared-buffer contract, so this module doesn't
re-express it as a value type. ``Controls`` and ``SimTime`` are the small marshalled values that cross
component boundaries on the host.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp


@dataclass(slots=True)
class SimTime:
    """Sim-time and step index for the current tick.

    The single authoritative time source for ``HIL_*`` MAVLink timestamps and the sensors' due times,
    so every component shares one consistent clock derived from the simulation. The time is a whole
    number of nanoseconds, as gz-sim keeps it, so a time that falls on a tick compares exactly; a
    stage reads it in seconds.
    """

    time_ns: int = 0
    """Elapsed simulation time since reset, in whole nanoseconds."""
    step_index: int = 0
    """Zero-based index of the current simulation step (integer tick counter)."""

    @property
    def sim_time(self) -> float:
        """Elapsed simulation time since reset, in seconds."""
        return self.time_ns / 1e9

    @property
    def time_usec(self) -> int:
        """Simulation time in integer microseconds, the ``HIL_*`` timestamp unit."""
        return round(self.time_ns / 1e3)


@wp.struct
class PoseTwist:
    """The pose and twist of one body in world axes: what the plant writes per body, as the signal
    ``body/<label>`` of shape ``(1,)``, and the estimate an estimator writes, as the signal ``estimate``,
    which the guidance and the controllers read.
    """

    position: wp.vec3
    """The body's origin, metres."""
    orientation: wp.quat
    """The body's orientation, a quaternion in ``(x, y, z, w)`` order."""
    linear_velocity: wp.vec3
    """The velocity of the body's center of mass, m/s."""
    angular_velocity: wp.vec3
    """The body's angular velocity, rad/s."""


@wp.struct
class ImuSample:
    """The sample of an Inertial Measurement Unit (IMU), in the axes of its mount: body
    Forward Right Down (FRD) on an unturned mount. The signal ``imu`` carries it, of shape ``(1,)``.
    """

    time: wp.float64
    """The sim time of the sample, seconds."""
    accel: wp.vec3
    """The specific force, m/s^2."""
    gyro: wp.vec3
    """The angular rate, rad/s."""


@wp.struct
class MagSample:
    """A magnetometer's sample, in body Forward Right Down (FRD) axes. The signal ``mag`` carries it, of
    shape ``(1,)``.
    """

    time: wp.float64
    """The sim time of the sample, seconds."""
    field: wp.vec3
    """The magnetic field, gauss."""


@wp.struct
class BaroSample:
    """A barometer's sample. The signal ``baro`` carries it, of shape ``(1,)``."""

    time: wp.float64
    """The sim time of the sample, seconds."""
    pressure: wp.float32
    """The static pressure, hPa."""
    altitude: wp.float32
    """The pressure altitude, metres."""
    temperature: wp.float32
    """The sensor's temperature, degrees Celsius."""


@wp.struct
class GpsSample:
    """A Global Positioning System (GPS) receiver's sample, on WGS84. The signal ``gps`` carries it, of shape
    ``(1,)``. Latitude, longitude and altitude are 64-bit: a 32-bit latitude loses about a metre.
    """

    time: wp.float64
    """The sim time of the sample, seconds."""
    lat: wp.float64
    """The latitude, degrees."""
    lon: wp.float64
    """The longitude, degrees."""
    alt: wp.float64
    """The altitude over mean sea level, metres."""
    velocity: wp.vec3
    """The velocity in North East Down (NED) axes, m/s."""
    ground_speed: wp.float32
    """The horizontal speed, m/s."""
    fix_type: wp.int32
    """The fix, as MAVLink's ``GPS_FIX_TYPE`` numbers it: 3 is a 3D fix."""


@dataclass(slots=True)
class Image:
    """A camera's frame, the host signal a camera writes when a frame arrives."""

    time: float
    """The sim time the frame shows, seconds."""
    pixels: np.ndarray
    """The image, ``(height, width, 3)`` 8-bit RGB."""


@dataclass(slots=True)
class PointCloud:
    """A lidar's scan, the host signal a lidar writes when a scan arrives."""

    time: float
    """The sim time the scan shows, seconds."""
    points: np.ndarray
    """The points, ``(n, 3)``, in the frame the Kit peer sends: world axes."""


# --- Setpoint: the guidance→controller command vocabulary ----------------
#
# Each setpoint travels as the signal `setpoint`, which a guidance writes and a controller reads: a position
# goal as one `wp.vec3`, a reference trajectory as itself on the host, so the builder checks that the two
# agree before the capture. A guidance writes it in place between graph replays, never inside the captured
# region, so the next replay reads it with zero re-capture.


@dataclass(slots=True)
class PositionGoal:
    """A single move-to / hold goal in the world frame, Newton FLU / Z-up. Consumed by the
    state-feedback controllers, policy and pid: the guidance feeds one ``PositionGoal`` at a time and
    sequences a mission by advancing it on arrival; the controller is goal-relative, so each is a
    fresh single-goal problem. ``yaw`` is the optional heading [rad]; ``None`` = don't command yaw.

    The signal ``setpoint`` carries it on the device as one ``wp.vec3`` of the position, since no
    controller reads yaw.
    """

    pos: tuple[float, float, float]
    yaw: float | None = None


@dataclass(slots=True)
class Waypoints:
    """An ordered list of world-frame positions the *controller* holds and advances internally.
    Distinct from a mission the *guidance* sequences with ``PositionGoal``: here the whole path is the
    setpoint.
    """

    points: list[tuple[float, float, float]] = field(default_factory=list)


@dataclass(slots=True)
class ReferenceTrajectory:
    """A full flat-state reference over time, for the acados NMPC: the guidance plans it, ruckig →
    differential-flatness, from waypoints and feeds it to the controller, which tracks it. The
    payload is a queryable reference object, for example the guidance's ``FlatnessReference``, the
    controller samples per horizon node, kept opaque here so core imports no controller backend.
    """

    reference: Any = None


# The neutral union of the setpoint types: a guidance writes one of them, and its controller declares
# the one it reads. Evaluated eagerly, as a real ``types.UnionType``, because the
# ``from __future__`` import only stringifies *annotations*, not this assignment, so it stays a usable
# runtime value: ``isinstance(sp, Setpoint)`` / ``typing.get_args(Setpoint)``.
Setpoint = PositionGoal | Waypoints | ReferenceTrajectory


def as_position_goal(sp) -> PositionGoal:
    """A :class:`PositionGoal` from what a caller passed: the goal itself, or a bare ``(x, y, z)``
    position, so a script can pass plain waypoints.

    Raises:
        TypeError: ``sp`` is neither a ``PositionGoal`` nor three numbers.
    """
    if isinstance(sp, PositionGoal):
        return sp
    arr = np.asarray(sp, dtype=float).reshape(-1)
    if arr.shape[0] != 3:
        raise TypeError(f"expected a PositionGoal or an (x, y, z) position, got {sp!r}")
    return PositionGoal(pos=(float(arr[0]), float(arr[1]), float(arr[2])))
