"""TrackingGuidance: plan one reference over a mission's whole path and hand it to a tracking controller."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from nexus._src.core.schema import ReferenceTrajectory, Setpoint

from .base import Guidance

if TYPE_CHECKING:
    from nexus._src.core.interfaces import Controller

_DISPLAY_REACHED_M = 0.3  # the marker turns green this close to a waypoint the reference flies through


class TrackingGuidance(Guidance):
    """Plan the whole-path reference once, on the first tick, and hand it to the controller.

    The planner takes the mission's waypoints and returns a reference that exposes ``set_start``,
    ``duration`` and ``reference_path``. The guidance plans on its first tick, which gives it the start
    position, hands the controller a ``ReferenceTrajectory`` before that tick's exchange,
    and ends the run with ``stop`` once the trajectory's duration plus ``final_hold_s`` has passed.
    ``reference_started_at`` is the sim time of that hand-over, the reference clock's anchor, which
    post-run evaluation reads.

    Args:
        controller: The tracking controller; it must expose ``accept_setpoint(ReferenceTrajectory)``.
        planner: ``waypoints -> reference``, the flight file's own, acados' min-snap planner.
        final_hold_s: Keep running this long, in sim time, after the trajectory completes.
        stop: A no-arg callable that ends the run, ``Orchestrator.stop``; ``None`` runs to ``max_steps``.
        body_index: The vehicle body whose position is the reference's start; 0 is the base.
    """

    def __init__(
        self,
        controller: Controller,
        *,
        planner: Callable,
        final_hold_s: float = 2.0,
        stop: Callable[[], None] | None = None,
        body_index: int = 0,
    ):
        super().__init__(controller, body_index=body_index)
        self._planner = planner
        self._final_hold_s = float(final_hold_s)
        self._stop = stop
        self._stop_at: float | None = None
        self._done = False
        self._planned = False
        self.reference_started_at: float | None = None

    def set_mission(self, setpoints: list[Setpoint]) -> None:
        """Set the waypoints the reference flies through; the plan waits for the first tick.

        Args:
            setpoints: Ordered waypoints, each a :class:`PositionGoal` or a bare ``(x, y, z)`` position.
                Must be non-empty.

        Raises:
            ValueError: ``setpoints`` is empty.
        """
        mission = [self._as_position_goal(s) for s in setpoints]
        if not mission:
            raise ValueError("set_mission needs at least one setpoint")
        self._mission = mission
        self._active = 0
        self._stop_at = None
        self._done = False
        self._planned = False
        self.reference_started_at = None

    def _tick(self, pos: np.ndarray, ts: float) -> None:
        if self._done:
            return
        if not self._planned:
            # Plan once, now that this tick gives the start position, and hand the reference over before this
            # tick's exchange; then end the run after the trajectory plus the settle.
            self._planned = True
            ref = self._planner([tuple(g.pos) for g in self._mission])
            ref.set_start(pos)
            self._controller.accept_setpoint(ReferenceTrajectory(reference=ref))
            self.reference_started_at = ts
            self._stop_at = ts + float(ref.duration) + self._final_hold_s  # max_steps still caps it
            self._emit(ref)
            return
        if ts >= self._stop_at:
            self._done = True
            self._active = len(self._mission)
            self._emit()  # recolor any marker the flight never came close to
            if self._stop is not None:
                self._stop()
            return
        # The reference flies through the waypoints in order: recolor each marker as the vehicle passes it.
        # Display only; the controller tracks the whole-path reference, and the run's end is time-based.
        if self._active < len(self._mission):
            active = np.asarray(self._mission[self._active].pos, dtype=float)
            if np.linalg.norm(pos - active) < _DISPLAY_REACHED_M:
                self._active += 1
                self._emit()
