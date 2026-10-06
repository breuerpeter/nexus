"""TrackingGuidance: plan one reference over a mission's whole path for a tracking controller."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from nexus._src.core.interfaces import Stage
from nexus._src.core.schema import ReferenceTrajectory, Setpoint

from .base import Guidance

_DISPLAY_REACHED_M = 0.3  # the marker turns green this close to a waypoint the reference flies through


class TrackingGuidance(Guidance):
    """Plan the whole-path reference once, on the run's first tick, and write it to the tick.

    The planner takes the mission's waypoints and returns a reference that exposes ``set_start``,
    ``duration`` and ``reference_path``. The guidance plans on the run's first tick, which gives it the
    start position, and writes a ``ReferenceTrajectory`` to that tick, which the loop hands to the
    controller before that tick's exchange. It marks the tick done once the trajectory's duration plus
    ``final_hold_s`` has passed, which ends the run. ``reference_started_at`` is the sim time of that
    hand-over, the reference clock's anchor, which post-run evaluation reads.

    Args:
        planner: ``waypoints -> reference``, the flight file's own, acados' min-snap planner.
        final_hold_s: Keep running this long, in sim time, after the trajectory completes.
        body_index: The vehicle body whose position is the reference's start; 0 is the base.
    """

    def __init__(self, *, planner: Callable, final_hold_s: float = 2.0, body_index: int = 0):
        super().__init__(body_index=body_index)
        self._planner = planner
        self._final_hold_s = float(final_hold_s)
        self._end_at: float | None = None
        self._planned = False
        self.reference_started_at: float | None = None

    def stages(self) -> list[Stage]:
        """One host stage, ``guidance``, which stays out of the warm pass. The controller starts the
        reference's clock at the exchange after the hand-over, and ``reference_started_at`` names that
        tick, so the plan waits for the run's first tick.
        """
        return [Stage("guidance", "host", self._run, warm=False)]

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
        self._end_at = None
        self._done = False
        self._planned = False
        self.reference_started_at = None

    def _tick(self, pos: np.ndarray, ts: float) -> None:
        if self._done:
            return
        if not self._planned:
            # Plan once, now that this tick gives the start position, and write the reference to this tick, so
            # the controller holds it at this tick's exchange; the mission is over after the trajectory plus
            # the settle.
            self._planned = True
            ref = self._planner([tuple(g.pos) for g in self._mission])
            ref.set_start(pos)
            self._command(ReferenceTrajectory(reference=ref))
            self.reference_started_at = ts
            self._end_at = ts + float(ref.duration) + self._final_hold_s  # max_steps still caps it
            self._emit(ref)
            return
        if ts >= self._end_at:
            self._done = True
            self._active = len(self._mission)
            self._emit()  # recolor any marker the flight never came close to
            return
        # The reference flies through the waypoints in order: recolor each marker as the vehicle passes it.
        # Display only; the controller tracks the whole-path reference, and the run's end is time-based.
        if self._active < len(self._mission):
            active = np.asarray(self._mission[self._active].pos, dtype=float)
            if np.linalg.norm(pos - active) < _DISPLAY_REACHED_M:
                self._active += 1
                self._emit()
