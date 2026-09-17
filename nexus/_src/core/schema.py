"""The neutral, typed vocabulary that flows between components, as architecture.md §1 describes.

A *logical* schema independent of any array library. For the eager slice the
hot-loop state is the physics backend's live ``newton.State``, typed under ``TYPE_CHECKING``
so core never imports it at runtime; the shared ``state.body_f`` device buffer
realizes ``Wrench``, see the shared-buffer contract, so this module doesn't re-express it
as a value type. ``Controls``/``Measurement``/``SimTime``/
``EnvSample`` are the small marshalled values that cross component boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import newton


@dataclass(slots=True)
class SimTime:
    """Sim-time and step index for the current tick.

    The single authoritative time source for ``HIL_*`` MAVLink timestamps,
    so every component shares one consistent clock derived from the simulation.
    """

    sim_time: float = 0.0
    """Elapsed simulation time since reset, in seconds."""
    step_index: int = 0
    """Zero-based index of the current simulation step (integer tick counter)."""

    @property
    def time_usec(self) -> int:
        """Simulation time in integer microseconds, the ``HIL_*`` timestamp unit."""
        return int(self.sim_time * 1e6)


@dataclass(slots=True)
class EnvSample:
    """Authoritative ambient fields at a point/time, as architecture.md §11 describes.

    v1 is a constant provider lifting the bridge's hardcoded gravity + World Magnetic Model (WMM) field.
    The world frame is Newton Forward Left Up (FLU) / Z-up.
    """

    gravity_world: tuple[float, float, float] = (0.0, 0.0, -9.81)  # Newton FLU/Z-up
    """Gravitational acceleration vector in the WORLD frame, m/s^2.

    World frame is Newton FLU / right-handed Z-up, so nominal gravity points along
    ``-Z`` (the ``(0, 0, -9.81)`` default).
    """
    # Earth magnetic field components in North East Down (NED) [gauss], the WMM sample at the origin.
    mag_ned: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Earth magnetic field components in the NED frame, gauss (WMM sample at the origin)."""
    air_pressure_msl: float = 1013.25  # [hPa]
    """Air pressure at mean sea level, hPa."""
    temperature: float = 25.0  # [degC]
    """Ambient air temperature, degrees Celsius."""


@dataclass(slots=True)
class Controls:
    """The controller→actuator command: always one command per actuator, normalized to ``[0, 1]``.

    This is the standardized actuator seam: the control law and the vehicle's mixer, the rate loop and
    the ``B⁻¹`` allocation, live in the controller, so whatever the law computes, Collective Thrust Body
    Rate (CTBR), moments or Nonlinear Model Predictive Control (NMPC) thrusts, ``command`` is what the
    controller emits and the actuator chain takes in. A length-``n`` host ``np.ndarray`` on the PX4 and
    eager deploy paths, or a device-native ``(1, n)`` Warp array in the captured in-process loop.
    """

    command: Any = None  # one entry per actuator: an np.ndarray of length n, or a (1, n) Warp array


@dataclass(slots=True)
class Measurement:
    """Per-tick sensor bundle in the Forward Right Down (FRD) body frame / physical units.

    For the slice the sensors fill one shared Measurement in place, the Hardware In The Loop (HIL)
    bundle PX4 consumes; the controller serializes it to MAVLink wire units.
    A field left at its default is simply one no sensor overrode this tick.
    """

    # --- Inertial Measurement Unit (IMU), HIL_SENSOR, body FRD ---
    xacc: float = 0.0
    """Specific force (accelerometer) along body FRD X (forward), m/s^2."""
    yacc: float = 0.0
    """Specific force (accelerometer) along body FRD Y (right), m/s^2."""
    zacc: float = 0.0  # specific force [m/s^2]
    """Specific force (accelerometer) along body FRD Z (down), m/s^2."""
    xgyro: float = 0.0
    """Angular rate about body FRD X (roll axis), rad/s."""
    ygyro: float = 0.0
    """Angular rate about body FRD Y (pitch axis), rad/s."""
    zgyro: float = 0.0  # [rad/s]
    """Angular rate about body FRD Z (yaw axis), rad/s."""
    # --- Magnetometer, HIL_SENSOR, body FRD [gauss] ---
    xmag: float = 0.0
    """Magnetic field along body FRD X (forward), gauss."""
    ymag: float = 0.0
    """Magnetic field along body FRD Y (right), gauss."""
    zmag: float = 0.0
    """Magnetic field along body FRD Z (down), gauss."""
    # --- Barometer, HIL_SENSOR ---
    abs_pressure: float = 1013.25  # [hPa]
    """Absolute (static) barometric pressure, hPa."""
    pressure_alt: float = 0.0  # [m]
    """Barometric pressure altitude, metres."""
    temperature: float = 25.0  # [degC]
    """Sensor temperature, degrees Celsius."""
    # --- Global Positioning System (GPS), HIL_GPS, physical units ---
    gps_valid: bool = False
    """Whether the GPS fields hold a valid fix this tick."""
    lat_deg: float = 0.0
    """WGS84 latitude, degrees."""
    lon_deg: float = 0.0
    """WGS84 longitude, degrees."""
    alt_m: float = 0.0  # altitude over mean sea level
    """Altitude above mean sea level (AMSL), metres."""
    vn: float = 0.0
    """GPS velocity north component (NED frame), m/s."""
    ve: float = 0.0
    """GPS velocity east component (NED frame), m/s."""
    vd: float = 0.0  # NED velocity [m/s]
    """GPS velocity down component (NED frame), m/s."""
    ground_speed: float = 0.0  # [m/s]
    """Horizontal ground speed, m/s."""
    fix_type: int = 3
    """GPS fix type (MAVLink ``GPS_FIX_TYPE``; 3 = 3D fix)."""
    eph: float = 1.0
    """Horizontal position dilution of precision (dimensionless)."""
    epv: float = 1.0
    """Vertical position dilution of precision (dimensionless)."""
    satellites: int = 10
    """Number of satellites visible/used in the solution."""
    # --- Attitude: HIL_STATE_QUATERNION, ground-truth visualization ---
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Ground-truth body attitude quaternion in **WXYZ** order (MAVLink convention).

    Body-to-world (NED) rotation as carried in ``HIL_STATE_QUATERNION``, used for
    visualization. Note the WXYZ order here, distinct from the XYZW order pinned on
    the device-native state path.
    """
    rollspeed: float = 0.0
    """Ground-truth roll rate about the body X axis, rad/s."""
    pitchspeed: float = 0.0
    """Ground-truth pitch rate about the body Y axis, rad/s."""
    yawspeed: float = 0.0
    """Ground-truth yaw rate about the body Z axis, rad/s."""
    # --- Perfect ground-truth kinematics for state-feedback consumers, cat-2 ---
    # An optional slot a ground-truth Sensor fills, for example the RL policy observation, so a
    # trained-policy Controller consumes it through the same exchange(meas) path as PX4.
    # Left None when no such sensor runs; never serialized to MAVLink.
    observation: Any = None
    """Optional ground-truth observation for state-feedback consumers (cat-2).

    A free-form slot a ground-truth ``Sensor`` fills (e.g. an RL policy observation
    vector) so a trained-policy ``Controller`` reads it through the same
    ``exchange(meas)`` path as PX4. ``None`` when no such sensor runs; never
    serialized to MAVLink.
    """
    # Ground-truth live newton.State for privileged state-feedback controllers; for example the
    # Model Predictive Control (MPC) seeds its planner from body_q/body_qd. StateSensor fills it; None
    # otherwise; never serialized.
    state: newton.State | None = None


# --- Setpoint: the operator→controller command vocabulary ----------------
#
# Setpoints are heterogeneous, so the core owns NO fixed buffer. A `Setpoint` is a small marshalled
# *intent* value, such as `Controls`/`Measurement`; the **controller** owns its own typed persistent
# buffer and writes it in place from the setpoint in ``Controller.accept_setpoint(sp)``, a §6
# value-mutation, so the next CUDA-graph replay picks it up with zero re-capture, per the capture contract.
# The operator, `InProcessOperator`, only flips that buffer, between graph replays, never inside the
# captured region. Each controller narrows the union to the variant it supports and raises on the rest.


@dataclass(slots=True)
class PositionGoal:
    """A single move-to / hold goal in the world frame, Newton FLU / Z-up. Consumed by the
    state-feedback controllers, policy and pid: the operator feeds one ``PositionGoal`` at a time and
    sequences a mission by advancing it on arrival; the controller is goal-relative, so each is a
    fresh single-goal problem. ``yaw`` is the optional heading [rad]; ``None`` = don't command yaw.
    """

    pos: tuple[float, float, float]
    yaw: float | None = None


@dataclass(slots=True)
class Waypoints:
    """An ordered list of world-frame positions the *controller* holds and advances internally;
    in the sampling MPC, reach-radius advances the persistent ``target`` buffer. Distinct from a
    mission the *operator* sequences with ``PositionGoal``: here the whole path is the setpoint.
    """

    points: list[tuple[float, float, float]] = field(default_factory=list)


@dataclass(slots=True)
class ReferenceTrajectory:
    """A full flat-state reference over time, for the acados NMPC: the operator plans it, ruckig →
    differential-flatness, from waypoints and feeds it to the controller, which tracks it. The
    payload is a queryable reference object, for example the operator's ``FlatnessReference``, the
    controller samples per horizon node, kept opaque here so core imports no controller backend.
    """

    reference: Any = None


# The neutral union every operator `goto`/`set_mission` carries and every controller's
# `accept_setpoint` narrows. Evaluated eagerly, as a real ``types.UnionType``, because the
# ``from __future__`` import only stringifies *annotations*, not this assignment, so it stays a usable
# runtime value: ``isinstance(sp, Setpoint)`` / ``typing.get_args(Setpoint)``.
Setpoint = PositionGoal | Waypoints | ReferenceTrajectory
