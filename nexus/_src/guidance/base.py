"""The guidance base: the mission a guidance holds, its stage, and the markers it logs.

A guidance is a component of the loop for a controller that takes setpoints. It states one host
stage, ``guidance``, which the loop runs before the controller's stages, so a setpoint written on a
tick is the one the controller reads on that tick. The stage runs between graph replays and only
writes the controller's setpoint buffer in place, through ``accept_setpoint``, so a captured graph
stays valid, per the capture contract. Logging is component-owned: the orchestrator hands over the
Logger, and the guidance emits its ``guidance/`` markers when the mission changes, on set, advance
or plan. Rerun shows the last value per entity path, so logging only on change is enough.
"""

from __future__ import annotations

import numpy as np

from nexus._src.core.interfaces import Controller, Stage
from nexus._src.core.schema import PositionGoal, as_position_goal

# guidance/ waypoint markers: the active goal is gold, a reached goal turns green, a future one is dim.
_GOLD = (255, 215, 0)
_GREEN = (60, 220, 110)
_PENDING = (120, 120, 130)
_RADIUS = 0.18  # sphere radius [m]


class Guidance:
    """Shared guidance state: the controller it commands, the mission, the stage and the markers.

    Args:
        controller: The controller that takes setpoints; it must expose ``accept_setpoint(Setpoint)``.
        body_index: The vehicle body whose position the guidance reads; 0 is the base.

    Raises:
        TypeError: ``controller`` has no ``accept_setpoint``.
    """

    def __init__(self, controller: Controller, *, body_index: int = 0):
        if not hasattr(controller, "accept_setpoint"):
            raise TypeError(
                f"{type(controller).__name__} has no accept_setpoint: a guidance commands a controller that "
                "takes setpoints, and PX4 flies its own missions over MAVLink"
            )
        self._controller = controller
        self._body_index = int(body_index)
        self._mission: list[PositionGoal] = []
        self._active: int = 0
        # Component-owned logging: the orchestrator hands over the Logger, None when off, which is the
        # gate, and sets the timeline at tick start, so an emitter just hands data to the Logger.
        self._logger = None

    # -- the stage, the seam the loop calls ------------------------------------------
    def stages(self) -> list[Stage]:
        """One host stage, ``guidance``, which the loop runs before the controller's stages."""
        return [Stage("guidance", "host", self._run)]

    def _run(self, tick) -> None:
        # One host copy of the body pose per tick, trivial next to a controller's host stage.
        pos = tick.state.body_q.numpy()[self._body_index][:3].astype(float)
        self._tick(pos, float(tick.t.sim_time))

    def _tick(self, pos: np.ndarray, ts: float) -> None:
        """One tick's work over the vehicle's world position and the sim time: each guidance's own."""
        raise NotImplementedError

    # -- setpoint normalization: accept a PositionGoal or a bare position ---------
    _as_position_goal = staticmethod(as_position_goal)

    # -- component-owned logging ---------------------------------------------------
    def set_logger(self, logger) -> None:
        """The orchestrator hands over the Logger, ``None`` when off: the gate for the guidance/ markers."""
        self._logger = logger

    def _emit(self, ref=None) -> None:
        """Emit the markers now, after a mission change: every waypoint, recolored by progress, plus the
        tracked reference when the caller passes one, after planning. A no-op when not recording.
        """
        if self._logger is None:
            return
        self._log_mission()
        if ref is not None:
            self._log_reference(ref)

    def _log_mission(self) -> None:
        """Emit every mission waypoint at once, reached → green, active → gold, future → dim, so the whole
        mission is visible; markers recolor as the mission advances, re-emitted on each change.
        """
        for i, g in enumerate(self._mission):
            color = _GREEN if i < self._active else (_GOLD if i == self._active else _PENDING)
            self._logger.log_points(
                f"guidance/waypoints/wp_{i}", [list(map(float, g.pos))], colors=color, radii=_RADIUS
            )

    def _log_reference(self, ref) -> None:
        """Emit the planned reference path a tracking controller follows, as a line strip under
        ``guidance/reference``. ``ref`` exposes ``reference_path()``, sampled only here, so a run that
        doesn't record never touches it.
        """
        pts = [[float(p[0]), float(p[1]), float(p[2])] for p in ref.reference_path()]
        self._logger.log_strip("guidance/reference", pts, color=(90, 160, 255), radius=0.025)
