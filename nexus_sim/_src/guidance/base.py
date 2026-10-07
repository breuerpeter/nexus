"""The guidance base: the mission a guidance holds, its stage, and the markers it logs.

A guidance is a component of the loop for a controller that reads a setpoint. It states one host
stage, ``guidance``, which the loop runs before the controller's stages. It holds no controller and no
stop. The stage writes a changed setpoint to the signal ``setpoint``, which the controller reads, in
place between graph replays, so a captured graph stays valid, per the capture contract, and the
controller reads the setpoint on that tick. The stage sets ``Tick.done`` when the mission is over, and
the loop ends the run. Logging is component-owned: the orchestrator hands over the Logger scoped to the
guidance's path, and the guidance emits its own rows, ``waypoints/wp_<i>`` and ``reference``, when the
mission changes, on set, advance or plan. Rerun shows the last value per entity path, so logging only on
change is enough.
"""

from __future__ import annotations

import numpy as np

from nexus_sim._src.core.interfaces import Stage, Tick
from nexus_sim._src.core.schema import PositionGoal, Setpoint, as_position_goal
from nexus_sim._src.core.signals import Signal

# The waypoint markers: the active goal is gold, a reached goal turns green, a future one is dim.
_GOLD = (255, 215, 0)
_GREEN = (60, 220, 110)
_PENDING = (120, 120, 130)
_RADIUS = 0.18  # sphere radius [m]


class Guidance:
    """Shared guidance state: the mission, the stage, its two outputs and the markers.

    A subclass does its work in ``_tick``. It commands a setpoint with ``_command`` and ends its mission
    by setting ``_done``. The stage writes the setpoint to its signal and marks the tick done.

    Args:
        setpoint: The signal ``setpoint`` the guidance writes, of the setpoint type its controller reads.
        body_index: The vehicle body whose position the guidance reads; 0 is the base.
    """

    def __init__(self, *, setpoint: Signal, body_index: int = 0):
        self.setpoint = setpoint
        self._body_index = int(body_index)
        self._mission: list[PositionGoal] = []
        self._active: int = 0
        # The two outputs: a setpoint commanded since the stage last ran, which the stage writes to the
        # signal, and whether the mission is over.
        self._commanded: Setpoint | None = None
        self._done = False
        # Component-owned logging: the orchestrator hands over the Logger, None when off, which is the
        # gate, and sets the timeline at tick start, so an emitter just hands data to the Logger.
        self._logger = None

    # -- the stage, the seam the loop calls ------------------------------------------
    def stages(self) -> list[Stage]:
        """One host stage, ``guidance``, which the loop runs before the controller's stages and which
        writes the setpoint. A warm stage: the loop also runs it once over the settled state, before any
        tick, so the controller holds the first setpoint before its own first stage.
        """
        return [Stage("guidance", "host", self._run, writes=(self.setpoint,))]

    def _run(self, tick: Tick) -> None:
        # One host copy of the body pose per tick, trivial next to a controller's host stage.
        pos = tick.state.body_q.numpy()[self._body_index][:3].astype(float)
        self._tick(pos, float(tick.t.sim_time))
        if self._commanded is not None:
            self._write(self._commanded)
            self._commanded = None
        if self._done:
            tick.done = True  # the loop ends the run after this tick

    def _tick(self, pos: np.ndarray, ts: float) -> None:
        """One tick's work over the vehicle's world position and the sim time: each guidance's own."""
        raise NotImplementedError

    def _command(self, setpoint: Setpoint) -> None:
        """Command a setpoint. The stage writes it to the signal on its next run, before the controller's
        stages. A later command replaces one the stage hasn't written yet.
        """
        self._commanded = setpoint

    def _write(self, setpoint: Setpoint) -> None:
        """Write a commanded setpoint to the signal: a host setpoint unchanged."""
        self.setpoint.write(setpoint)

    # -- setpoint normalization: accept a PositionGoal or a bare position ---------
    _as_position_goal = staticmethod(as_position_goal)

    # -- component-owned logging ---------------------------------------------------
    def set_logger(self, logger) -> None:
        """The orchestrator hands over the Logger, scoped to the guidance's path, or ``None`` when off: the
        gate for the markers. Each emitter names only its own row.
        """
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
            self._logger.log_points(f"waypoints/wp_{i}", [list(map(float, g.pos))], colors=color, radii=_RADIUS)

    def _log_reference(self, ref) -> None:
        """Emit the planned reference path a tracking controller follows, as a line strip at the
        guidance's row ``reference``. ``ref`` exposes ``reference_path()``, sampled only here, so a run that
        doesn't record never touches it.
        """
        pts = [[float(p[0]), float(p[1]), float(p[2])] for p in ref.reference_path()]
        self._logger.log_strip("reference", pts, color=(90, 160, 255), radius=0.025)
