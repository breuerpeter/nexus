"""MissionGuidance: sequence a mission of position goals for a controller that takes setpoints."""

from __future__ import annotations

import numpy as np

from nexus._src.core.schema import PositionGoal, Setpoint

from .base import Guidance


class MissionGuidance(Guidance):
    """Sequence a mission of position goals: advance on arrival, and end the run after the final hold.

    The stage runs each tick before the controller's: it detects arrival at the active goal, writes the
    next goal to the tick, which the loop hands to the controller, and marks the tick done once the
    final goal has held for ``final_hold_s``, which ends the run. The stage is a warm stage, so the
    controller holds the first goal before its own first stage; a mission of two or more goals advances
    between graph replays.

    A subclass, such as the policy example's geofence, overrides ``_tick``, the stage's work over the
    vehicle's position and the sim time. It can command a setpoint with ``_command``, read and set
    ``_done``, which freezes the sequencing and ends the run, log through ``_logger``, and read
    ``_body_index``. The rest is this class's own.

    Args:
        reached_m: Advance to the next goal within this distance [m] of the active one.
        final_hold_s: Keep running this long, in sim time, after the vehicle reaches the final goal.
        body_index: The vehicle body whose position drives arrival detection; 0 is the base.
    """

    def __init__(self, *, reached_m: float = 0.3, final_hold_s: float = 2.0, body_index: int = 0):
        super().__init__(body_index=body_index)
        self._reached_m = float(reached_m)
        self._final_hold_s = float(final_hold_s)
        self._end_at: float | None = None  # sim time at which the mission is over, set on the final goal
        self._announced = False  # log the first gold marker on the stage's first run, with a valid timeline
        # Mission telemetry, which post-run evaluation reads: the sim time the vehicle reached each goal.
        self.arrival_times: list[float] = []

    def goto(self, setpoint: Setpoint) -> None:
        """Issue one setpoint, a single-goal mission.

        Args:
            setpoint: The target: a :class:`PositionGoal` or a bare ``(x, y, z)`` position.
        """
        self.set_mission([setpoint])

    def set_mission(self, setpoints: list[Setpoint]) -> None:
        """Set the mission; the guidance advances through it on arrival and owns the run's end.

        Call before ``sim.run()``: the stage writes the first goal to the tick on its next run, the warm
        pass of a run that hasn't started, and the guidance advances on arrival.

        Args:
            setpoints: Ordered mission setpoints, each a :class:`PositionGoal` or a bare ``(x, y, z)``
                position. Must be non-empty.

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
        self._announced = False
        self.arrival_times = []
        self._command(mission[0])

    def _tick(self, pos: np.ndarray, ts: float) -> None:
        if self._done:
            return
        if not self._announced:  # first run: show the whole mission, active gold, future dim
            self._announced = True
            self._emit()
        if self._end_at is not None:  # final goal reached: hold a beat, then the mission is over
            if ts >= self._end_at:
                self._done = True
            return
        if self._advance_on_arrival(pos, ts):
            if self._active < len(self._mission):
                self._command(self._mission[self._active])
            else:
                self._end_at = ts + self._final_hold_s

    def _advance_on_arrival(self, pos: np.ndarray, ts: float) -> bool:
        """Within ``reached_m`` of the active goal → record the arrival, advance, and recolor the markers:
        reached → green, next → gold.

        Returns:
            ``True`` on arrival.
        """
        if self._active >= len(self._mission):
            return False
        active = self._mission[self._active]
        if np.linalg.norm(pos - np.asarray(active.pos, dtype=float)) >= self._reached_m:
            return False
        self.arrival_times.append(ts)
        self._active += 1
        self._emit()
        return True

    @property
    def reached(self) -> int:
        """How many mission goals the vehicle has reached, for a post-run assertion."""
        return min(self._active, len(self._mission))

    @property
    def active_setpoint(self) -> PositionGoal | None:
        """The active commanded goal, or ``None`` until ``set_mission`` runs or once the vehicle has
        reached every goal.
        """
        return self._mission[self._active] if self._mission and self._active < len(self._mission) else None
