"""MissionGuidance and TrackingGuidance: the mission a guidance sequences or plans, driven through its
stages, the seam the loop calls. A guidance holds no controller and no stop: its stage writes a
changed setpoint to the tick and marks the tick done when the mission is over. A stand-in state holds
the vehicle's position: no real sim.
"""

import numpy as np
import pytest

pytest.importorskip("warp")

import warp as wp

from nexus_sim._src.core.interfaces import Tick
from nexus_sim._src.core.schema import PositionGoal, ReferenceTrajectory, SimTime
from nexus_sim._src.guidance import MissionGuidance, TrackingGuidance
from nexus_sim.examples.controllers.policy.goto.geofence import GeofenceGuidance


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


def _tick(guidance, pos, sim_time) -> Tick:
    """Run the guidance's stages once, as the loop does on one tick, and return that tick."""
    tick = Tick(state=_State(pos), t=SimTime(sim_time, 0), dt=0.004, meas=None)
    for stage in guidance.stages():
        stage.run(tick)
    return tick


def test_a_guidance_constructs_from_its_own_parameters():
    """A guidance constructs from its own parameters: no controller, no orchestrator and no stop."""
    built = [
        MissionGuidance(reached_m=0.3),
        TrackingGuidance(planner=_Reference),
        GeofenceGuidance(bounds=((-3.0, -3.0, 0.2), (3.0, 3.0, 4.0))),
    ]
    assert [type(guidance).__name__ for guidance in built] == [
        "MissionGuidance",
        "TrackingGuidance",
        "GeofenceGuidance",
    ]


def test_a_mission_guidance_states_one_warm_host_stage():
    """A mission guidance's stage runs in the warm pass, so the controller holds the first goal before
    its own first stage.
    """
    assert [(st.name, st.kind, st.warm) for st in MissionGuidance().stages()] == [("guidance", "host", True)]


def test_a_tracking_guidance_states_one_host_stage_that_waits_for_the_first_tick():
    """A tracking guidance's stage stays out of the warm pass, so its plan waits for the run's first tick."""
    stages = TrackingGuidance(planner=_Reference).stages()
    assert [(st.name, st.kind, st.warm) for st in stages] == [("guidance", "host", False)]


def test_a_guidance_writes_a_changed_setpoint_to_the_tick():
    """A guidance writes a changed setpoint to the tick.

    Given a `MissionGuidance` with two goals, when its stage runs on a tick where the vehicle is within
    reach of the first goal, then the tick's setpoint is the second goal, and when it runs on a tick
    with no arrival, then it leaves the tick's setpoint empty.
    """
    guidance = MissionGuidance(reached_m=0.3)
    guidance.set_mission([(1.0, 0.0, 2.0), (0.0, 1.0, 3.0)])
    first = _tick(guidance, (0.0, 0.0, 0.0), 0.0).setpoint  # the stage's first run writes the first goal
    idle = _tick(guidance, (0.0, 0.0, 0.0), 0.1).setpoint
    arrived = _tick(guidance, (1.0, 0.0, 2.0), 1.0).setpoint
    assert (first, idle, arrived) == (PositionGoal(pos=(1.0, 0.0, 2.0)), None, PositionGoal(pos=(0.0, 1.0, 3.0)))


def test_set_mission_makes_the_first_goal_the_active_one():
    guidance = MissionGuidance()
    guidance.set_mission([(1.0, 0.0, 2.0), (0.0, 1.0, 3.0)])
    assert (guidance.active_setpoint, guidance.reached) == (PositionGoal(pos=(1.0, 0.0, 2.0)), 0)


def test_an_empty_mission_raises():
    with pytest.raises(ValueError):
        MissionGuidance().set_mission([])


def test_the_mission_advances_on_arrival_and_ends_after_the_final_hold():
    guidance = MissionGuidance(reached_m=0.3, final_hold_s=2.0)
    guidance.set_mission([(1.0, 0.0, 2.0), (0.0, 1.0, 3.0)])

    # far from wp0 → no advance
    tick = _tick(guidance, (0.0, 0.0, 0.0), 0.1)
    assert (guidance.reached, tick.done) == (0, False)

    # at wp0 → advance to wp1, which the stage writes to the tick
    tick = _tick(guidance, (1.0, 0.0, 2.0), 1.0)
    assert (guidance.reached, tick.setpoint) == (1, PositionGoal(pos=(0.0, 1.0, 3.0)))
    assert guidance.arrival_times == [1.0]

    # at wp1, the final one → no goal beyond the last, and the hold starts
    tick = _tick(guidance, (0.0, 1.0, 3.0), 5.0)
    assert (guidance.reached, tick.setpoint, tick.done) == (2, None, False)

    # before the hold elapses: the mission goes on
    assert _tick(guidance, (0.0, 1.0, 3.0), 6.5).done is False
    # after final_hold_s (5.0 + 2.0): the mission is over, and the stage marks the tick done
    assert _tick(guidance, (0.0, 1.0, 3.0), 7.0).done is True
    assert _tick(guidance, (0.0, 1.0, 3.0), 8.0).done is True  # a mission that's over marks every tick


def test_goto_is_a_single_goal_mission():
    guidance = MissionGuidance()
    guidance.goto(PositionGoal(pos=(2.0, 2.0, 2.0)))
    assert _tick(guidance, (0.0, 0.0, 0.0), 0.0).setpoint == PositionGoal(pos=(2.0, 2.0, 2.0))


def test_a_tracking_guidance_plans_and_writes_a_reference_once_then_ends():
    planned = {}

    def planner(waypoints):
        planned["waypoints"] = waypoints
        return _Reference(waypoints)

    guidance = TrackingGuidance(planner=planner, final_hold_s=2.0)
    guidance.set_mission([(2.0, 0.5, 3.5), (3.0, 2.0, 4.0)])
    assert planned == {}  # nothing planned yet: the plan waits for the first tick, which gives the start position

    start = (0.0, 0.0, 2.0)
    tick = _tick(guidance, start, 0.004)
    assert isinstance(tick.setpoint, ReferenceTrajectory)  # the guidance planned, and wrote the reference
    assert planned["waypoints"] == [(2.0, 0.5, 3.5), (3.0, 2.0, 4.0)]
    np.testing.assert_allclose(tick.setpoint.reference.start, [0.0, 0.0, 2.0])
    assert guidance.reference_started_at == 0.004

    tick = _tick(guidance, start, 1.0)  # mid-trajectory: no new plan, and the mission goes on
    assert (tick.setpoint, tick.done) == (None, False)
    assert _tick(guidance, start, 5.5).done is False  # before 0.004 + duration(4.0) + hold(2.0)
    assert _tick(guidance, start, 6.5).done is True  # past it → the mission is over
