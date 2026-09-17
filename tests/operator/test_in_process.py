"""InProcessOperator: mission sequencing at the host seam. Advance on arrival, accept_setpoint, stop.

Exercises the operator wiring with a fake controller, which records accept_setpoint, and a fake state
with a settable position: no real sim. The operator's `tick` is the orchestrator's on_tick seam.
"""

import numpy as np
import pytest

from nexus._src.core.schema import PositionGoal
from nexus._src.operator import InProcessOperator


class _FakeController:
    def __init__(self):
        self.setpoints: list[PositionGoal] = []

    def accept_setpoint(self, sp):
        assert isinstance(sp, PositionGoal)
        self.setpoints.append(sp)


class _FakeState:
    """State-like: body_q rows [p, q_xyzw]; the operator reads body_q.numpy()[i][:3]."""

    def __init__(self):
        self.pos = np.zeros(3)

    @property
    def body_q(self):
        import warp as wp

        return wp.array(np.concatenate([self.pos, [0, 0, 0, 1]]).reshape(1, 7).astype(np.float32), dtype=wp.transform)


class _T:
    def __init__(self, sim_time):
        self.sim_time = sim_time


class _FakeTrackingController:
    """A tracking controller, acados-shaped: accepts a ReferenceTrajectory, not a PositionGoal."""

    def __init__(self):
        self.references = []

    def accept_setpoint(self, sp):
        from nexus._src.core.schema import ReferenceTrajectory

        assert isinstance(sp, ReferenceTrajectory)
        self.references.append(sp.reference)


class _FakeReference:
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.start = None
        self.duration = 4.0

    def set_start(self, p0):
        self.start = np.asarray(p0, dtype=float)


def test_rejects_a_controller_without_accept_setpoint():
    with pytest.raises(TypeError):
        InProcessOperator(object())


def test_planner_mode_plans_and_hands_a_reference_once_then_ends():
    ctrl = _FakeTrackingController()
    planned = {}

    def planner(waypoints):
        planned["waypoints"] = waypoints
        return _FakeReference(waypoints)

    stops = {"n": 0}
    op = InProcessOperator(ctrl, planner=planner, final_hold_s=2.0, stop=lambda: stops.__setitem__("n", stops["n"] + 1))
    view = _FakeState()
    op.set_mission([(2.0, 0.5, 3.5), (3.0, 2.0, 4.0)])
    assert ctrl.references == []  # nothing handed yet: planned on the first tick, which needs the start position

    view.pos = np.array([0.0, 0.0, 2.0])  # the start position
    op.tick(view, _T(0.004), 1)
    assert len(ctrl.references) == 1  # the operator planned + handed the reference once
    assert planned["waypoints"] == [(2.0, 0.5, 3.5), (3.0, 2.0, 4.0)]
    np.testing.assert_allclose(ctrl.references[0].start, [0.0, 0.0, 2.0])

    op.tick(view, _T(1.0), 2)  # mid-trajectory: no re-plan, no stop
    assert len(ctrl.references) == 1 and stops["n"] == 0
    op.tick(view, _T(5.5), 3)  # before duration(4.0)+hold(2.0)=6.0 → still running
    assert stops["n"] == 0
    op.tick(view, _T(6.5), 4)  # after 6.0 → end the run once
    assert stops["n"] == 1


def test_set_mission_commands_the_first_goal_immediately():
    ctrl = _FakeController()
    op = InProcessOperator(ctrl)
    op.set_mission([(1.0, 0.0, 2.0), (0.0, 1.0, 3.0)])
    assert ctrl.setpoints == [PositionGoal(pos=(1.0, 0.0, 2.0))]  # first goal set before the run
    assert op.active_setpoint == PositionGoal(pos=(1.0, 0.0, 2.0))
    assert op.reached == 0


def test_empty_mission_raises():
    with pytest.raises(ValueError):
        InProcessOperator(_FakeController()).set_mission([])


def test_advances_on_arrival_and_stops_after_final_hold():
    ctrl = _FakeController()
    stops = {"n": 0}
    op = InProcessOperator(ctrl, reached_m=0.3, final_hold_s=2.0, stop=lambda: stops.__setitem__("n", stops["n"] + 1))
    view = _FakeState()
    op.set_mission([(1.0, 0.0, 2.0), (0.0, 1.0, 3.0)])

    # far from wp0 → no advance
    view.pos = np.array([0.0, 0.0, 0.0])
    op.tick(view, _T(0.1), 1)
    assert op.reached == 0
    assert len(ctrl.setpoints) == 1

    # at wp0 → advance to wp1, accept_setpoint called again
    view.pos = np.array([1.0, 0.0, 2.0])
    op.tick(view, _T(1.0), 2)
    assert op.reached == 1
    assert ctrl.setpoints[-1] == PositionGoal(pos=(0.0, 1.0, 3.0))

    # at wp1, the final one → no further setpoint; schedule the stop after the hold
    view.pos = np.array([0.0, 1.0, 3.0])
    op.tick(view, _T(5.0), 3)
    assert op.reached == 2
    assert len(ctrl.setpoints) == 2  # no goal beyond the last
    assert stops["n"] == 0  # not yet: within the hold

    # before the hold elapses: still no stop
    op.tick(view, _T(6.5), 4)
    assert stops["n"] == 0
    # after final_hold_s (5.0 + 2.0): stop fires exactly once
    op.tick(view, _T(7.0), 5)
    assert stops["n"] == 1
    op.tick(view, _T(8.0), 6)  # idempotent: done, no second stop
    assert stops["n"] == 1


def test_goto_is_a_single_goal_mission():
    ctrl = _FakeController()
    op = InProcessOperator(ctrl)
    op.goto(PositionGoal(pos=(2.0, 2.0, 2.0)))
    assert ctrl.setpoints == [PositionGoal(pos=(2.0, 2.0, 2.0))]


def test_px4_only_verbs_raise_for_in_process():
    op = InProcessOperator(_FakeController())
    assert op.is_armed() is True
    for verb in (lambda: op.set_mode("Hold"), lambda: op.takeoff(5.0), op.land):
        with pytest.raises(NotImplementedError):
            verb()
