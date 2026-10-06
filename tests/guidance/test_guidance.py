"""MissionGuidance and TrackingGuidance: the mission a guidance sequences or plans, driven through its
stages, the seam the loop calls. A stand-in controller keeps each setpoint it receives, and a stand-in
state holds the vehicle's position: no real sim.
"""

import numpy as np
import pytest

pytest.importorskip("warp")

import warp as wp

from nexus._src.core.interfaces import Tick
from nexus._src.core.schema import PositionGoal, ReferenceTrajectory, SimTime
from nexus._src.guidance import MissionGuidance, TrackingGuidance


class _Controller:
    """A controller that takes position goals, and keeps each one it receives."""

    def __init__(self):
        self.setpoints: list[PositionGoal] = []

    def accept_setpoint(self, sp):
        assert isinstance(sp, PositionGoal)
        self.setpoints.append(sp)


class _TrackingController:
    """A tracking controller, the acados shape: it takes a reference trajectory, not a position goal."""

    def __init__(self):
        self.references = []

    def accept_setpoint(self, sp):
        assert isinstance(sp, ReferenceTrajectory)
        self.references.append(sp.reference)


class _Reference:
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.start = None
        self.duration = 4.0

    def set_start(self, p0):
        self.start = np.asarray(p0, dtype=float)


class _State:
    """The physics state a guidance reads: one body at a set position."""

    def __init__(self, pos=(0.0, 0.0, 0.0)):
        self.pos = pos

    @property
    def body_q(self):
        return wp.array(np.array([[*self.pos, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32), dtype=wp.transform)


def _tick(guidance, pos, sim_time):
    """Run the guidance's stages once, as the loop does on one tick."""
    tick = Tick(state=_State(pos), t=SimTime(sim_time, 0), dt=0.004, meas=None)
    for stage in guidance.stages():
        stage.run(tick)


def test_a_guidance_rejects_a_controller_that_takes_no_setpoints():
    with pytest.raises(TypeError):
        MissionGuidance(object())


def test_a_guidance_states_one_host_stage():
    assert [(st.name, st.kind) for st in MissionGuidance(_Controller()).stages()] == [("guidance", "host")]


def test_set_mission_commands_the_first_goal_at_once():
    ctrl = _Controller()
    guidance = MissionGuidance(ctrl)
    guidance.set_mission([(1.0, 0.0, 2.0), (0.0, 1.0, 3.0)])
    assert ctrl.setpoints == [PositionGoal(pos=(1.0, 0.0, 2.0))]  # first goal set before the run
    assert guidance.active_setpoint == PositionGoal(pos=(1.0, 0.0, 2.0))
    assert guidance.reached == 0


def test_an_empty_mission_raises():
    with pytest.raises(ValueError):
        MissionGuidance(_Controller()).set_mission([])


def test_the_mission_advances_on_arrival_and_the_run_ends_after_the_final_hold():
    ctrl = _Controller()
    stops = {"n": 0}
    guidance = MissionGuidance(
        ctrl, reached_m=0.3, final_hold_s=2.0, stop=lambda: stops.__setitem__("n", stops["n"] + 1)
    )
    guidance.set_mission([(1.0, 0.0, 2.0), (0.0, 1.0, 3.0)])

    # far from wp0 → no advance
    _tick(guidance, (0.0, 0.0, 0.0), 0.1)
    assert guidance.reached == 0
    assert len(ctrl.setpoints) == 1

    # at wp0 → advance to wp1, accept_setpoint called again
    _tick(guidance, (1.0, 0.0, 2.0), 1.0)
    assert guidance.reached == 1
    assert ctrl.setpoints[-1] == PositionGoal(pos=(0.0, 1.0, 3.0))
    assert guidance.arrival_times == [1.0]

    # at wp1, the final one → no further setpoint; schedule the stop after the hold
    _tick(guidance, (0.0, 1.0, 3.0), 5.0)
    assert guidance.reached == 2
    assert len(ctrl.setpoints) == 2  # no goal beyond the last
    assert stops["n"] == 0  # not yet: within the hold

    # before the hold elapses: still no stop
    _tick(guidance, (0.0, 1.0, 3.0), 6.5)
    assert stops["n"] == 0
    # after final_hold_s (5.0 + 2.0): stop fires exactly once
    _tick(guidance, (0.0, 1.0, 3.0), 7.0)
    assert stops["n"] == 1
    _tick(guidance, (0.0, 1.0, 3.0), 8.0)  # idempotent: done, no second stop
    assert stops["n"] == 1


def test_goto_is_a_single_goal_mission():
    ctrl = _Controller()
    guidance = MissionGuidance(ctrl)
    guidance.goto(PositionGoal(pos=(2.0, 2.0, 2.0)))
    assert ctrl.setpoints == [PositionGoal(pos=(2.0, 2.0, 2.0))]


def test_a_tracking_guidance_plans_and_hands_a_reference_once_then_ends():
    ctrl = _TrackingController()
    planned = {}

    def planner(waypoints):
        planned["waypoints"] = waypoints
        return _Reference(waypoints)

    stops = {"n": 0}
    guidance = TrackingGuidance(
        ctrl, planner=planner, final_hold_s=2.0, stop=lambda: stops.__setitem__("n", stops["n"] + 1)
    )
    guidance.set_mission([(2.0, 0.5, 3.5), (3.0, 2.0, 4.0)])
    assert ctrl.references == []  # nothing handed yet: planned on the first tick, which needs the start position

    start = (0.0, 0.0, 2.0)
    _tick(guidance, start, 0.004)
    assert len(ctrl.references) == 1  # the guidance planned + handed the reference once
    assert planned["waypoints"] == [(2.0, 0.5, 3.5), (3.0, 2.0, 4.0)]
    np.testing.assert_allclose(ctrl.references[0].start, [0.0, 0.0, 2.0])
    assert guidance.reference_started_at == 0.004

    _tick(guidance, start, 1.0)  # mid-trajectory: no re-plan, no stop
    assert len(ctrl.references) == 1 and stops["n"] == 0
    _tick(guidance, start, 5.5)  # before 0.004 + duration(4.0) + hold(2.0) → still running
    assert stops["n"] == 0
    _tick(guidance, start, 6.5)  # past it → end the run once
    assert stops["n"] == 1
