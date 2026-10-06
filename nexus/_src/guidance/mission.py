"""MissionGuidance: sequence a mission of position goals for a controller that takes setpoints."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from nexus._src.core.schema import PositionGoal, Setpoint

from .base import Guidance

if TYPE_CHECKING:
    from nexus._src.core.interfaces import Controller


class MissionGuidance(Guidance):
    """Sequence a mission of position goals: advance on arrival, and end the run after the final hold.

    The stage runs each tick before the controller's: it detects arrival at the active goal, hands the
    controller the next one through ``accept_setpoint``, and ends the run with ``stop`` once the final
    goal has held for ``final_hold_s``. The guidance writes a single goal once, before the run; a
    multi-goal mission advances between graph replays.

    A subclass, such as the policy example's geofence, can read and set ``_done``, which freezes the
    sequencing, call ``_stop``, log through ``_logger``, and read ``_body_index``. The rest is this
    class's own.

    Args:
        controller: The controller that takes setpoints; it must expose ``accept_setpoint(Setpoint)``.
        reached_m: Advance to the next goal within this distance [m] of the active one.
        final_hold_s: Keep running this long, in sim time, after the vehicle reaches the final goal.
        stop: A no-arg callable that ends the run, ``Orchestrator.stop``; ``None`` runs to ``max_steps``.
        body_index: The vehicle body whose position drives arrival detection; 0 is the base.
    """

    def __init__(
        self,
        controller: Controller,
        *,
        reached_m: float = 0.3,
        final_hold_s: float = 2.0,
        stop: Callable[[], None] | None = None,
        body_index: int = 0,
    ):
        super().__init__(controller, body_index=body_index)
        self._reached_m = float(reached_m)
        self._final_hold_s = float(final_hold_s)
        self._stop = stop
        self._stop_at: float | None = None  # sim time to end the run, armed on the final goal
        self._done = False
        self._announced = False  # log the first gold marker on the first tick, with a valid timeline
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

        Call before ``sim.run()``: the controller gets the first goal at once, and the guidance advances
        on arrival.

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
        self._stop_at = None
        self._done = False
        self._announced = False
        self.arrival_times = []
        self._controller.accept_setpoint(mission[0])  # command the first goal in place, captured once

    def _tick(self, pos: np.ndarray, ts: float) -> None:
        if self._done:
            return
        if not self._announced:  # first tick: show the whole mission, active gold, future dim
            self._announced = True
            self._emit()
        if self._stop_at is not None:  # final goal reached: hold a beat, then end the run
            if ts >= self._stop_at:
                self._done = True
                self._end()
            return
        if self._advance_on_arrival(pos, ts):
            if self._active < len(self._mission):
                self._controller.accept_setpoint(self._mission[self._active])  # in-place, between replays
            else:
                self._stop_at = ts + self._final_hold_s

    def _end(self) -> None:
        """End the run through ``stop``, when the flight handed one over."""
        if self._stop is not None:
            self._stop()

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
