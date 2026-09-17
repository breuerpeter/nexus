"""InProcessOperator: command an in-process autopilot, policy / pid / mpc, by flipping its setpoint.

The in-process operator runs in the same process as the controller but **outside the captured hot
loop**: ``goto`` / ``set_mission`` call ``controller.accept_setpoint(sp)``, a §6 value-mutation of
the controller's persistent buffer, and the mission advances at the orchestrator's **host seam**
between graph replays; :meth:`tick` hooks into ``Orchestrator.on_tick``, the post-step observer,
the exact slot a recorder/``on_tick`` already uses. So a single fixed goal set before the run gets
captured once, with zero per-tick host op; a multi-waypoint mission advances between replays. Nothing
ever mutates inside the captured region, so capture/determinism hold, per the capture contract.

This relocates the old example ``on_tick``, waypoint advance + stop + sphere logging, into the
operator, so the example becomes ``set_mission`` + ``run`` + an assertion, per architecture.md §10:
component-owned logging; the example carries none.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from nexus._src.core.schema import PositionGoal, Setpoint

from .operator import BaseOperator

if TYPE_CHECKING:
    from nexus._src.core.interfaces import Controller

# A verb this command source doesn't model: an in-process autopilot has no arming/RC/mode plane;
# it flies on accept_setpoint. Px4Offboard models the full PX4 plane.
_PX4_ONLY = "{verb} is a PX4 operator verb; an in-process autopilot flies via goto/set_mission"


class InProcessOperator(BaseOperator):
    """Sequence a mission for an in-process controller, via ``accept_setpoint``.

    Args:
        controller: the in-process autopilot; must expose ``accept_setpoint(Setpoint)``.
        reached_m: advance to the next goal within this distance [m] of the active one.
        final_hold_s: keep running this long, in sim-time, after the final goal / the trajectory completes.
        stop: a no-arg callable that ends the run, ``Orchestrator.stop``; ``None`` = run to max_steps.
        body_index: the vehicle body whose position drives arrival detection; 0 = base.
        planner: optional ``waypoints -> reference`` factory for a TRACKING controller, acados. When
            given, ``set_mission`` doesn't sequence ``PositionGoal``s; the operator plans the whole-path
            reference once, on the first tick, with the start position, and hands the controller a
            ``ReferenceTrajectory``, then ends the run after the trajectory duration + ``final_hold_s``.
    """

    def __init__(
        self,
        controller: Controller,
        *,
        reached_m: float = 0.3,
        final_hold_s: float = 2.0,
        stop: Callable[[], None] | None = None,
        body_index: int = 0,
        planner: Callable | None = None,
    ):
        super().__init__()
        if not hasattr(controller, "accept_setpoint"):
            raise TypeError(
                f"{type(controller).__name__} has no accept_setpoint: it is not an in-process controller "
                "(PX4 is commanded over MAVLink via Px4Offboard, not via InProcessOperator)"
            )
        self._controller = controller
        self._reached_m = float(reached_m)
        self._final_hold_s = float(final_hold_s)
        self._stop = stop
        self._body_index = int(body_index)
        self._planner = planner  # set → trajectory-tracking mode: plan + hand a ReferenceTrajectory once
        self._planned = False
        self._stop_at: float | None = None  # sim-time to stop after the final goal / the trajectory
        self._done = False
        self._announced = False  # log the first gold marker on tick 0, with a valid timeline ts
        # Mission telemetry, which post-run evaluation reads: the sim time the vehicle reached each waypoint,
        # and, in tracking mode, the sim time the operator handed the planned reference to the controller,
        # the reference clock's anchor: reference time = sim time − reference_started_at.
        self.arrival_times: list[float] = []
        self.reference_started_at: float | None = None

    # -- operator verbs -----------------------------------------------------------
    def goto(self, setpoint: Setpoint) -> None:
        """Issue one setpoint, a single-goal "mission."

        Args:
            setpoint: The target: a :class:`PositionGoal` or a bare ``(x, y, z)`` position.
        """
        self.set_mission([setpoint])

    def set_mission(self, setpoints: list[Setpoint]) -> None:
        """Set the mission; the operator advances through it on arrival and owns the run's end.

        Call before ``sim.run()``. A goal-relative controller, policy/pid/sampling-mpc, gets the first
        ``PositionGoal`` immediately and the operator advances on arrival; a tracking controller, acados
        with ``planner`` set, gets a whole-path ``ReferenceTrajectory`` planned once on the first tick instead.

        Args:
            setpoints: Ordered mission setpoints; each a :class:`PositionGoal` or a bare
                ``(x, y, z)`` position. Must be non-empty.

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
        self._planned = False
        self.arrival_times = []
        self.reference_started_at = None
        if self._planner is not None:
            return  # tracking mode: plan + hand the reference on the first tick; needs the start position
        self._controller.accept_setpoint(mission[0])  # command the first goal in place, captured once

    # -- host seam: advance the mission between graph replays, Orchestrator.on_tick --
    def tick(self, state, t: float, steps: int) -> None:
        """Post-step observer, output-only + setpoint value-mutation: detect arrival at the active
        goal, advance to the next via ``accept_setpoint``, and stop the run after the final hold.

        Hooked into ``Orchestrator.on_tick``, the host seam between graph replays, so its only
        side effects are output, the ``/operator`` markers, and the §6 value-mutation of the
        controller's setpoint buffer; never a mutation inside the captured region.

        Args:
            state (newton.State): The live physics state; reads the vehicle's current pose this tick.
            t: The current sim time [s]; drives arrival detection and the final-hold timing.
            steps: The control-step count so far; unused, part of the ``on_tick`` signature.
        """
        if self._done:
            return
        ts = float(getattr(t, "sim_time", t))
        pos = state.body_q.numpy()[self._body_index][:3].astype(float)  # world-frame body origin
        if self._planner is not None:
            # Tracking mode, acados: plan the whole-path reference once, now that the start position is
            # available, and hand it to the controller; then end the run after the trajectory + settle.
            if not self._planned:
                from nexus._src.core.schema import ReferenceTrajectory

                self._planned = True
                ref = self._planner([tuple(g.pos) for g in self._mission])
                ref.set_start(pos)
                self._controller.accept_setpoint(ReferenceTrajectory(reference=ref))
                self.reference_started_at = ts  # the reference clock's sim-time anchor; evaluation reads it
                self._stop_at = ts + float(ref.duration) + self._final_hold_s  # max_steps still caps it
                self._emit(ref)  # emit the mission + the ruckig/flatness reference, event-driven
            elif ts >= self._stop_at:
                self._done = True
                self._active = len(self._mission)
                self._emit()  # recolor any marker the flight never came within reached_m of
                if self._stop is not None:
                    self._stop()
            else:
                # The reference flies THROUGH the waypoints in order: the same arrival detection as the
                # sequencing mode, but display/telemetry-only; the controller tracks the whole-path
                # reference, and the run's end stays time-based.
                self._advance_on_arrival(pos, ts)
            return
        if not self._announced:  # first tick: show the whole mission, active gold, future dim
            self._announced = True
            self._emit()
        if self._stop_at is not None:  # final goal reached: hold a beat, then end the run
            if ts >= self._stop_at:
                self._done = True
                if self._stop is not None:
                    self._stop()
            return
        if self._advance_on_arrival(pos, ts):
            if self._active < len(self._mission):
                nxt = self._mission[self._active]
                self._controller.accept_setpoint(nxt)  # in-place value-mutation at the host seam
            else:
                self._stop_at = ts + self._final_hold_s

    def _advance_on_arrival(self, pos, ts: float) -> bool:
        """The one arrival detection, for both modes: within ``reached_m`` of the active goal → record the
        arrival, advance, and recolor the markers; reached → green, next → gold; event-driven.

        Returns:
            ``True`` on arrival; the caller applies its mode's consequences: the sequencing mode
            commands the next goal / arms the final hold; the tracking mode has none.
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
        """The active commanded goal, or ``None`` until ``set_mission`` runs."""
        return self._mission[self._active] if self._mission and self._active < len(self._mission) else None

    # -- telemetry: an in-process autopilot is "armed" whenever it's flying -------
    def is_armed(self) -> bool:
        """Always ``True``: an in-process autopilot is "armed" whenever it's flying."""
        return True

    def mode(self) -> str:
        """The fixed mode name ``"in-process"``; this command source has no mode plane."""
        return "in-process"

    def landed_state(self) -> str:
        """Always ``"IN_AIR"``: an in-process autopilot has no ground/landing state machine."""
        return "IN_AIR"

    # -- PX4-plane verbs an in-process autopilot doesn't model ---------------------
    def arm(self) -> bool:
        """No-op arm: an in-process autopilot flies via ``goto``/``set_mission``.

        Returns:
            ``True``; always armed when flying.
        """
        return True

    def disarm(self) -> bool:
        """No-op disarm: an in-process autopilot has no arming plane.

        Returns:
            ``True``.
        """
        return True

    def set_mode(self, mode: str) -> None:
        """Unsupported: an in-process autopilot has no mode plane.

        Raises:
            NotImplementedError: Always; ``set_mode`` is a PX4 operator verb.
        """
        raise NotImplementedError(_PX4_ONLY.format(verb="set_mode"))

    def takeoff(self, alt_m: float) -> None:
        """Unsupported: fly via ``goto``/``set_mission`` instead.

        Raises:
            NotImplementedError: Always; ``takeoff`` is a PX4 operator verb.
        """
        raise NotImplementedError(_PX4_ONLY.format(verb="takeoff"))

    def land(self) -> None:
        """Unsupported: an in-process autopilot has no land verb.

        Raises:
            NotImplementedError: Always; ``land`` is a PX4 operator verb.
        """
        raise NotImplementedError(_PX4_ONLY.format(verb="land"))

    def set_rc(self, *, roll: float = 0.0, pitch: float = 0.0, throttle: float = 0.5, yaw: float = 0.0) -> None:
        """Unsupported: an in-process autopilot has no RC, or manual-control, plane.

        Raises:
            NotImplementedError: Always; ``set_rc`` is a PX4 operator verb.
        """
        raise NotImplementedError(_PX4_ONLY.format(verb="set_rc"))
