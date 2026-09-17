"""The Operator plane, Plane 5, *who commands the vehicle*, and its shared base.

The operator reads input, a script/API now, a joystick, Human Interface Device (HID) or
Ground Control Station (GCS) later, and converts it to autopilot commands. It's **never a node
inside the captured graph**: it lives *outside* the device hot loop, as the command source. The
out-of-process implementations, ``Px4Offboard`` over MAVLink, are literally remote; the in-process
one, ``InProcessOperator``, runs in-process but still *between* graph replays: it only flips the
controller's setpoint buffer, which the in-loop controller reads. So capturability survives intact,
per the capture contract.

The protocol is synchronous/transport-agnostic, uniform with ``Sim``, with no asyncio imposed on
callers; ``Px4Offboard`` hides MAVLink's async/streaming nature behind a background thread.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np

from nexus._src.core.schema import PositionGoal, Setpoint

# /operator waypoint markers, relocated from the example as component-owned logging, architecture.md
# §10: the active goal is gold, a reached goal turns green. The next goal only shows on arrival.
_GOLD = (255, 215, 0)
_GREEN = (60, 220, 110)
_PENDING = (120, 120, 130)  # future, not-yet-active waypoints: dim
_RADIUS = 0.18  # sphere radius [m]


@runtime_checkable
class Operator(Protocol):
    """Controller-agnostic operator intent, Plane 5. Implementations differ only in **input source**
    and **output transport**; adding one leaves the rest of the system unchanged.
    """

    def arm(self) -> bool:
        """Request that the vehicle's motors arm. Returns once the request lodges, not once it
        has taken effect; observe that with :meth:`is_armed`.

        Returns:
            Implementation-specific and not a confirmation: ``InProcessOperator`` returns ``True``,
            since it's always armed when flying; ``Px4Offboard`` returns nothing.
        """
        ...

    def disarm(self) -> bool:
        """Disarm the vehicle's motors.

        Returns:
            ``True`` once the operator has issued the disarm command.
        """
        ...

    def set_mode(self, mode: str) -> None:
        """Request the named flight mode. Returns once the request lodges, not once the autopilot
        has entered the mode; observe that with :meth:`mode`.

        Args:
            mode: Implementation-specific flight-mode name, for example PX4's ``"Takeoff"``,
                ``"Hold"``, ``"Land"``.
        """
        ...

    def takeoff(self, alt_m: float) -> None:
        """Take off and climb to the given altitude.

        Args:
            alt_m: Target altitude over the launch point, in meters.
        """
        ...

    def land(self) -> None:
        """Land the vehicle in place."""
        ...

    def goto(self, setpoint: Setpoint) -> None:
        """Command a single setpoint, a one-element mission.

        Args:
            setpoint: The target: a :class:`PositionGoal`, ``Waypoints``, or
                ``ReferenceTrajectory``, or a bare ``(x, y, z)`` position.
        """
        ...

    def set_mission(self, setpoints: list[Setpoint]) -> None:
        """Set the mission the operator flies, advancing through it over the run.

        Args:
            setpoints: Ordered setpoints; each a :class:`Setpoint` or a bare
                ``(x, y, z)`` position.
        """
        ...

    def set_rc(self, *, roll: float, pitch: float, throttle: float, yaw: float) -> None:
        """Set the manual-control, or stick, setpoint.

        Args:
            roll: Roll stick, in ``[-1, 1]``.
            pitch: Pitch stick, in ``[-1, 1]``.
            throttle: Throttle stick, in ``[0, 1]``.
            yaw: Yaw stick, in ``[-1, 1]``.
        """
        ...

    def is_armed(self) -> bool:
        """Report the vehicle's armed state.

        Returns:
            ``True`` if armed; an in-process autopilot is "armed" whenever it's flying.
        """
        ...

    def mode(self) -> str:
        """Report the current flight mode.

        Returns:
            The current mode name, for example ``"Takeoff"``; ``"in-process"`` for an
            in-process autopilot.
        """
        ...

    def landed_state(self) -> str:
        """Report the landed state.

        Returns:
            One of ``"ON_GROUND"`` / ``"IN_AIR"`` / ``"TAKING_OFF"`` / ``"LANDING"`` /
            ``"UNKNOWN"``.
        """
        ...


class BaseOperator:
    """Shared operator state: the active mission + ``/operator`` logging.

    Subclasses supply the **transport**: ``InProcessOperator`` flips the controller's setpoint
    buffer directly; ``Px4Offboard`` speaks MAVLink. The mission is a list of :class:`PositionGoal`;
    it also accepts raw 3-tuples / ``np.ndarray`` positions and normalizes them, so a script can pass
    bare waypoints. Logging is the component ``log(t, logger)`` step, which the orchestrator fans out only
    when a ``Logger`` is present; output-only, the same seam every component uses.
    """

    def __init__(self):
        self._mission: list[PositionGoal] = []
        self._active: int = 0
        # Component-owned logging: the orchestrator hands over the Logger, ``_logger``, None when off, which
        # is the gate, and sets the timeline at tick start. The operator emits its /operator viz event-driven:
        # :meth:`_emit` runs right when the mission display changes, on set, advance or plan. Rerun
        # shows the last value per entity path, so logging only on change is enough.
        self._logger = None

    def set_logger(self, logger) -> None:
        """The orchestrator hands over the Logger, ``None`` when off: the gate for the /operator viz."""
        self._logger = logger

    def _emit(self, ref=None) -> None:
        """Emit the /operator viz now, after a mission change: all markers, recolored by progress, plus
        the tracked reference when the caller passes one, after planning. No-op when not recording, meaning
        ``_logger is None``. The orchestrator already set the timeline this tick, so this just hands data
        to the Logger.
        """
        if self._logger is None:
            return
        self._log_mission()
        if ref is not None:
            self._log_reference(ref)

    # -- setpoint normalization: accept a PositionGoal or a bare position ---------
    @staticmethod
    def _as_position_goal(sp: Setpoint) -> PositionGoal:
        if isinstance(sp, PositionGoal):
            return sp
        arr = np.asarray(sp, dtype=float).reshape(-1)
        if arr.shape[0] != 3:
            raise TypeError(f"expected a PositionGoal or an (x, y, z) position, got {sp!r}")
        return PositionGoal(pos=(float(arr[0]), float(arr[1]), float(arr[2])))

    # -- /operator viz emitters: hand data to the Logger; the orchestrator has set the timeline --
    def _log_mission(self) -> None:
        """Emit every mission waypoint at once, reached → green, active → gold, future → dim, so the whole
        mission is visible; markers recolor as the mission advances, re-emitted on each change.
        """
        for i, g in enumerate(self._mission):
            color = _GREEN if i < self._active else (_GOLD if i == self._active else _PENDING)
            self._logger.log_points(
                f"operator/waypoints/wp_{i}", [list(map(float, g.pos))], colors=color, radii=_RADIUS
            )

    def _log_reference(self, ref) -> None:
        """Emit the planned reference path, the ruckig/flatness trajectory a TRACKING controller follows, as
        a LineStrip under ``operator/reference``. ``ref`` exposes ``reference_path()``, sampled only here, so
        a non-recording run never touches it.
        """
        pts = [[float(p[0]), float(p[1]), float(p[2])] for p in ref.reference_path()]
        self._logger.log_strip("operator/reference", pts, color=(90, 160, 255), radius=0.025)


def wait_until(predicate: Callable[[], bool], timeout: float, poll: float = 0.05) -> None:
    """Block until ``predicate()`` is true, or raise ``TimeoutError``.

    A wall-clock poll, so the predicate's source must refresh itself: it suits operator telemetry,
    say ``lambda: not operator.is_armed()``, which ``Px4Offboard``'s pump thread keeps fresh. A
    sim-state predicate needs ``Sim.wait_until``, which steps the sim; nothing else advances it.

    Args:
        predicate: Zero-arg callable polled until it returns true.
        timeout: Wall-clock seconds to wait before giving up.
        poll: Sleep interval in wall-clock seconds between polls.

    Raises:
        TimeoutError: ``predicate()`` didn't become true within ``timeout`` seconds.
    """
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return
        if time.monotonic() >= deadline:
            break
        time.sleep(poll)
    raise TimeoutError(f"condition not met within {timeout}s")
