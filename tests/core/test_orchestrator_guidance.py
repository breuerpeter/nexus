"""The loop runs a guidance's stage before the controller's, so the setpoint a guidance computes on a
tick is the one the controller reads on that tick. A guidance holds no controller and no stop: its
stage writes the setpoint signal the controller reads, and marks the tick done when its mission is
over, which the loop ends the run on. Stand-in components in the loop's roles, on the Warp CPU device,
where every stage runs as plain Python on each tick; a stand-in estimator hands the guidance the
scripted position.
"""

import logging

import pytest

pytest.importorskip("warp")

import warp as wp

from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.core.orchestrator import Orchestrator
from nexus_sim._src.core.schema import PoseTwist, ReferenceTrajectory, SimTime
from nexus_sim._src.core.signals import Signal
from nexus_sim._src.guidance import MissionGuidance, TrackingGuidance

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
    """The physics state: one body, at the position a test scripts."""

    def __init__(self, pos):
        self.pos = pos


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


class _Estimator:
    """An estimator whose stage writes the scripted position to the estimate the guidance reads."""

    capturable = True

    def __init__(self):
        self.estimate = Signal("estimate", PoseTwist, shape=(1,))

    def stages(self):
        def estimate(tick):
            self.estimate.write([(tuple(tick.state.pos), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))])

        return [Stage("estimate", "device", estimate, writes=(self.estimate,))]


class _Actuator:
    capturable = True

    def stages(self):
        return [Stage("forces", "device", lambda tick: None)]


class _Sensor:
    capturable = True

    def stages(self):
        return [Stage("sample", "device", lambda tick: None)]


class _ExchangeController:
    """A controller that reads a setpoint of type `setpoint` and solves on the host: a ``read`` and an
    ``exchange`` host stage. It keeps, per exchange, the sim time, the vehicle's position and the setpoint
    it reads.
    """

    def __init__(self, state, setpoint=wp.vec3):
        self._state = state
        self.setpoint = Signal("setpoint", setpoint, shape=(1,) if setpoint is wp.vec3 else None)
        self.exchanges = []

    def connect(self):
        pass

    def close(self):
        pass

    def stages(self):
        def exchange(tick):
            self.exchanges.append((tick.t.sim_time, tuple(self._state.pos), self.setpoint.read()))
            return True

        return [Stage("read", "host", lambda tick: None), Stage("exchange", "host", exchange, reads=(self.setpoint,))]


class _DeviceController:
    """A controller that reads a setpoint and whose stages are all device stages, the shape of the
    Proportional Integral Derivative (PID) example. It keeps the setpoint it reads each time its
    stage runs.
    """

    capturable = True

    def __init__(self):
        self.setpoint = Signal("setpoint", wp.vec3, shape=(1,))
        self.held = []

    def connect(self):
        pass

    def close(self):
        pass

    def stages(self):
        return [Stage("act", "device", lambda tick: self.held.append(self.setpoint.read()), reads=(self.setpoint,))]


class _NoSetpointController:
    """A controller that reads no setpoint, the shape of PX4's."""

    def connect(self):
        pass

    def close(self):
        pass

    def stages(self):
        return [Stage("read", "host", lambda tick: None), Stage("exchange", "host", lambda tick: True)]


class _Reference:
    """A planned reference, the shape a tracking guidance hands its controller."""

    duration = 4.0

    def set_start(self, p0):
        pass

    def reference_path(self):
        return []


class _Recording:
    """The recording sink, the shape the loop calls. It keeps the entity path of every row a component
    logs through the view the loop scopes for it.
    """

    log_interval = 0.0

    def __init__(self):
        self.entities = []

    def scoped(self, path):
        return _View(self, path)

    def set_time(self, t):
        pass

    def log_rtf(self, rtf):
        pass

    def log_profile(self, stats):
        pass

    def close(self):
        pass


class _View:
    """A component's view of the recording, as the logger's ``scoped`` hands one: its rows land under
    the path the loop asked for.
    """

    def __init__(self, recording, path):
        self._recording = recording
        self._path = path

    def log_points(self, name, positions, **style):
        self._recording.entities.append(f"{self._path}/{name}")

    def log_strip(self, name, points, **style):
        self._recording.entities.append(f"{self._path}/{name}")


def _orch(controller, physics, **kw):
    return Orchestrator(
        clock=_Clock(),
        physics=physics,
        actuator=_Actuator(),
        sensors=[_Sensor()],
        estimator=_Estimator(),
        controller=controller,
        **kw,
    )


def test_a_controller_reads_the_next_goal_on_the_tick_the_vehicle_arrives():
    """A controller reads the next goal on the tick the vehicle arrives: the guidance writes the setpoint
    before the controller's stages read it, and the guidance holds no controller.
    """
    with wp.ScopedDevice("cpu"):
        physics = _Physics([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), GOALS[0], GOALS[0]])
        controller = _ExchangeController(physics.state)
        guidance = MissionGuidance(reached_m=0.3)
        guidance.set_mission(GOALS)
        orch = _orch(controller, physics, guidance=guidance)
        for _ in range(3):
            orch.step()
        orch.close()
    held = next(setpoint for _, pos, setpoint in controller.exchanges if pos == GOALS[0])
    holds_the_controller = any(value is controller for value in vars(guidance).values())
    assert (tuple(float(v) for v in held[0]), holds_the_controller) == (GOALS[1], False)


def test_a_controller_holds_the_first_goal_before_its_own_first_stage():
    """A controller holds the first goal before its own first stage.

    Given a run whose controller records the goal it holds each time its device stage runs and a
    `MissionGuidance` that holds its mission before the run, when the run takes its first tick, then
    every run of that stage, the warm pass included, held the first goal.
    """
    with wp.ScopedDevice("cpu"):
        controller = _DeviceController()
        guidance = MissionGuidance()
        guidance.set_mission(GOALS)
        orch = _orch(controller, _Physics([(0.0, 0.0, 0.0)]), guidance=guidance)
        orch.step()
        orch.close()
    held = [tuple(float(v) for v in sp[0]) for sp in controller.held]
    # The stage runs once in the warm pass and once on the first tick.
    assert held == [GOALS[0], GOALS[0]]


def test_a_tracking_controller_holds_its_planned_reference_from_its_first_exchange():
    """A tracking controller holds its planned reference from its first exchange, and the plan waits
    for the run's first tick: the seed exchange, over the settled state, held no reference.
    """
    reference = _Reference()
    with wp.ScopedDevice("cpu"):
        physics = _Physics([(0.0, 0.0, 2.0)])
        controller = _ExchangeController(physics.state, ReferenceTrajectory)
        guidance = TrackingGuidance(planner=lambda waypoints: reference)
        guidance.set_mission([(2.0, 0.5, 3.5), (3.0, 2.0, 4.0)])
        orch = _orch(controller, physics, guidance=guidance)
        orch.step()  # the run's first tick; its exchange is the last one the controller kept
        orch.close()
    _, _, seeded = controller.exchanges[0]  # the seed exchange, before any tick
    sim_time, _, setpoint = controller.exchanges[-1]
    assert (seeded, setpoint.reference, guidance.reference_started_at) == (None, reference, sim_time)


def test_a_run_whose_stages_are_all_device_stages_stays_one_captured_segment_with_a_guidance(caplog):
    """A run whose stages are all device stages stays one captured segment with a guidance."""
    with wp.ScopedDevice("cpu"), caplog.at_level(logging.INFO, logger="nexus"):
        controller = _DeviceController()
        guidance = MissionGuidance()
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
        guidance = MissionGuidance(reached_m=0.3)
        guidance.set_mission(GOALS)
        orch = _orch(
            controller, physics, guidance=guidance, on_tick=lambda state, t, steps: ticks.append(steps), max_steps=6
        )
        while orch.step():
            pass
        orch.close()
    assert (len(ticks), guidance.reached) == (6, 2)


def test_a_mission_that_is_over_ends_the_run():
    """A mission that's over ends the run.

    Given a run whose `max_steps` lies far beyond its mission and a `MissionGuidance` with one goal and
    no final hold, when the vehicle reaches the goal, then `run()` returns, and the run took no tick
    after the one on which the mission ended.
    """
    ticks = []
    with wp.ScopedDevice("cpu"):
        physics = _Physics([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), GOALS[0]])
        controller = _ExchangeController(physics.state)
        guidance = MissionGuidance(reached_m=0.3, final_hold_s=0.0)
        guidance.set_mission(GOALS[:1])
        orch = _orch(
            controller, physics, guidance=guidance, on_tick=lambda state, t, steps: ticks.append(steps), max_steps=50
        )
        orch.run()
        orch.close()
    # The vehicle arrives on the second tick, and the stage ends the mission on the next one.
    assert (ticks, guidance.reached) == ([1, 2, 3], 1)


def test_a_run_whose_controller_takes_no_setpoint_refuses_a_guidance():
    """A run whose controller reads no setpoint refuses a guidance.

    Given an orchestrator whose controller reads no setpoint and a `MissionGuidance`, when the run takes
    its first step, then it raises `TypeError` that names the controller's class.
    """
    with wp.ScopedDevice("cpu"):
        guidance = MissionGuidance()
        guidance.set_mission(GOALS)
        orch = _orch(_NoSetpointController(), _Physics([(0.0, 0.0, 0.0)]), guidance=guidance)
        with pytest.raises(TypeError, match="_NoSetpointController"):
            orch.step()
        orch.close()


class _SetpointSensor:
    """A sensor whose device stage reads a signal named `setpoint`, as the guidance's is."""

    def __init__(self):
        self.setpoint = Signal("setpoint", wp.vec3, shape=(1,))

    def stages(self):
        return [Stage("sample", "device", lambda tick: None, reads=(self.setpoint,))]


def test_a_run_whose_controller_reads_no_setpoint_refuses_a_guidance_that_another_component_reads():
    """A run whose controller reads no setpoint refuses a guidance, even when another component reads it.

    Given a controller that reads no setpoint, a sensor that reads a signal named `setpoint` and a
    `MissionGuidance`, when the run takes its first step, then it raises `TypeError` that names the
    controller's class.
    """
    with wp.ScopedDevice("cpu"):
        guidance = MissionGuidance()
        guidance.set_mission(GOALS)
        orch = Orchestrator(
            clock=_Clock(),
            physics=_Physics([(0.0, 0.0, 0.0)]),
            actuator=_Actuator(),
            sensors=[_SetpointSensor()],
            estimator=_Estimator(),
            controller=_NoSetpointController(),
            guidance=guidance,
        )
        with pytest.raises(TypeError, match="_NoSetpointController"):
            orch.step()
        orch.close()


def test_the_mission_markers_and_the_tracked_reference_sit_under_guidance_in_the_recording():
    """The mission markers and the tracked reference sit under `guidance/` in the recording."""
    recording = _Recording()
    with wp.ScopedDevice("cpu"):
        physics = _Physics([(0.0, 0.0, 2.0)])
        controller = _ExchangeController(physics.state)
        mission = MissionGuidance()
        mission.set_mission([(1.0, 0.0, 2.0), (2.0, 0.0, 2.0), (3.0, 0.0, 2.0)])
        orch = _orch(controller, physics, guidance=mission, logger=recording)
        orch.step()
        orch.close()
        physics = _Physics([(0.0, 0.0, 2.0)])
        controller = _ExchangeController(physics.state, ReferenceTrajectory)
        tracking = TrackingGuidance(planner=lambda waypoints: _Reference())
        tracking.set_mission([(2.0, 0.5, 3.5), (3.0, 2.0, 4.0)])
        orch = _orch(controller, physics, guidance=tracking, logger=recording)
        orch.step()
        orch.close()
    assert sorted(set(recording.entities)) == [
        "guidance/reference",
        "guidance/waypoints/wp_0",
        "guidance/waypoints/wp_1",
        "guidance/waypoints/wp_2",
    ]
