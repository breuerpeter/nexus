"""The loop runs a guidance's stage before the controller's, so the setpoint a guidance computes on a
tick is the one the controller reads on that tick. Stand-in components at the loop's seams, on the
Warp CPU device, where every stage runs as plain Python on each tick.
"""

import logging

import numpy as np
import pytest

pytest.importorskip("warp")

import warp as wp

from nexus._src.core.interfaces import Stage
from nexus._src.core.orchestrator import Orchestrator
from nexus._src.core.schema import Controls, SimTime
from nexus._src.guidance import MissionGuidance, TrackingGuidance

GOALS = [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]


class _Clock:
    dt = 0.004

    def __init__(self):
        self._t = SimTime()

    def now(self):
        return self._t

    def advance(self):
        self._t = SimTime(self._t.sim_time + self.dt, self._t.step_index + 1)
        return self._t

    def throttle(self):
        pass


class _State:
    """The physics state a guidance reads: one body, at the position a test scripts."""

    def __init__(self, pos):
        self.pos = pos

    @property
    def body_q(self):
        return wp.array(np.array([[*self.pos, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32), dtype=wp.transform)


class _Physics:
    """Physics whose ``step`` moves the vehicle one point along a scripted path, and holds the last."""

    capturable = True

    def __init__(self, path):
        self._path = list(path)
        self._steps = 0
        self.state = _State(self._path[0])

    def reset(self):
        return self.state

    def clear_forces(self, state):
        pass

    def step(self, state, dt):
        self._steps += 1
        state.pos = self._path[min(self._steps, len(self._path) - 1)]
        return state

    def stages(self):
        return [
            Stage("clear", "device", lambda tick: self.clear_forces(tick.state)),
            Stage("step", "device", lambda tick: self.step(tick.state, tick.dt)),
        ]


class _Actuator:
    capturable = True

    def stages(self):
        return [Stage("forces", "device", lambda tick: None)]


class _Sensor:
    capturable = True

    def read(self, meas):
        pass

    def stages(self):
        return [Stage("sample", "device", lambda tick: None)]


class _ExchangeController:
    """A controller that takes setpoints and solves on the host: a ``read`` and an ``exchange`` host
    stage. It keeps, per exchange, the sim time, the vehicle's position and the setpoint it holds.
    """

    def __init__(self, state):
        self._state = state
        self.setpoint = None
        self.exchanges = []

    def connect(self):
        pass

    def close(self):
        pass

    def accept_setpoint(self, sp):
        self.setpoint = sp

    def stages(self):
        def exchange(tick):
            self.exchanges.append((tick.t.sim_time, tuple(self._state.pos), self.setpoint))
            tick.controls = Controls(command=[0.0, 0.0, 0.0, 0.0])
            return True

        return [Stage("read", "host", lambda tick: None), Stage("exchange", "host", exchange)]


class _DeviceController:
    """A controller that takes setpoints and whose stages are all device stages, the shape of the
    Proportional Integral Derivative (PID) example.
    """

    capturable = True

    def connect(self):
        pass

    def close(self):
        pass

    def accept_setpoint(self, sp):
        pass

    def stages(self):
        return [Stage("act", "device", lambda tick: None)]


class _Reference:
    """A planned reference, the shape a tracking guidance hands its controller."""

    duration = 4.0

    def set_start(self, p0):
        pass

    def reference_path(self):
        return []


def _orch(controller, physics, **kw):
    return Orchestrator(
        clock=_Clock(), physics=physics, actuator=_Actuator(), sensors=[_Sensor()], controller=controller, **kw
    )


def test_a_controller_reads_the_next_goal_on_the_tick_the_vehicle_arrives():
    """A controller reads the next goal on the tick the vehicle arrives."""
    with wp.ScopedDevice("cpu"):
        physics = _Physics([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), GOALS[0], GOALS[0]])
        controller = _ExchangeController(physics.state)
        guidance = MissionGuidance(controller, reached_m=0.3)
        guidance.set_mission(GOALS)
        orch = _orch(controller, physics, guidance=guidance)
        for _ in range(3):
            orch.step()
        orch.close()
    held = next(setpoint for _, pos, setpoint in controller.exchanges if pos == GOALS[0])
    assert tuple(held.pos) == GOALS[1]


def test_a_tracking_controller_holds_its_planned_reference_from_its_first_exchange():
    """A tracking controller holds its planned reference from its first exchange."""
    reference = _Reference()
    with wp.ScopedDevice("cpu"):
        physics = _Physics([(0.0, 0.0, 2.0)])
        controller = _ExchangeController(physics.state)
        guidance = TrackingGuidance(controller, planner=lambda waypoints: reference)
        guidance.set_mission([(2.0, 0.5, 3.5), (3.0, 2.0, 4.0)])
        orch = _orch(controller, physics, guidance=guidance)
        orch.step()  # the run's first tick; its exchange is the last one the controller kept
        orch.close()
    sim_time, _, setpoint = controller.exchanges[-1]
    assert (setpoint.reference, guidance.reference_started_at) == (reference, sim_time)


def test_a_run_whose_stages_are_all_device_stages_stays_one_captured_segment_with_a_guidance(caplog):
    """A run whose stages are all device stages stays one captured segment with a guidance."""
    with wp.ScopedDevice("cpu"), caplog.at_level(logging.INFO, logger="nexus"):
        controller = _DeviceController()
        guidance = MissionGuidance(controller)
        guidance.set_mission(GOALS)
        orch = _orch(controller, _Physics([(0.0, 0.0, 0.0)]), guidance=guidance)
        orch.step()
        orch.close()
    plan = next(m for m in (r.getMessage() for r in caplog.records) if m.startswith("stage plan:"))
    # A device segment reads `graph(...)` when it captures and `eager(...)` on the CPU device.
    assert (plan.count("graph(") + plan.count("eager("), plan.count("host(")) == (1, 1)


def test_a_runs_on_tick_hook_and_its_guidance_both_run():
    """A run's `on_tick` hook and its guidance both run."""
    ticks = []
    with wp.ScopedDevice("cpu"):
        physics = _Physics([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), GOALS[0], GOALS[1]])
        controller = _ExchangeController(physics.state)
        guidance = MissionGuidance(controller, reached_m=0.3)
        guidance.set_mission(GOALS)
        orch = _orch(
            controller, physics, guidance=guidance, on_tick=lambda state, t, steps: ticks.append(steps), max_steps=6
        )
        while orch.step():
            pass
        orch.close()
    assert (len(ticks), guidance.reached) == (6, 2)
