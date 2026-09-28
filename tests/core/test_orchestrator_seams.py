"""The seams the loop drives take no ambient sample: physics, the actuator and each sensor read the
state and the clock, and nothing else. Warp-free stubs, as in test_orchestrator_stop.py.
"""

from nexus._src.core.orchestrator import Orchestrator
from nexus._src.core.schema import Controls, SimTime


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


class _Physics:
    """Steps a one-number state: the actuator adds one, the step multiplies by ten, so the value the
    sensor reads on the next tick says which of the two ran, and in which order.
    """

    def reset(self):
        return {"q": 0}

    def clear_forces(self, state):
        pass

    def step(self, state, dt):
        state["q"] *= 10
        return state


class _Actuator:
    def forces(self, controls, state):
        state["q"] += 1


class _Sensor:
    """Copies the state into the measurement, so what the controller receives shows the sensor ran."""

    def sample(self, state, t, out):
        out.temperature = state["q"]


class _Controller:
    """Answers the preroll at once and keeps the last measurement it was handed."""

    def __init__(self):
        self.meas = None

    def connect(self):
        pass

    def exchange(self, meas, t, timeout):
        self.meas = meas
        return Controls(command=[0.0, 0.0, 0.0, 0.0])

    def close(self):
        pass


def test_the_loop_runs_a_tick_on_seams_that_take_no_env():
    """The loop runs a tick on seams that take no `env`: `Physics.step(state, dt)`,
    `Actuator.forces(controls, state)` and `Sensor.sample(state, t, out)`.
    """
    controller = _Controller()
    orch = Orchestrator(
        clock=_Clock(),
        physics=_Physics(),
        actuator=_Actuator(),
        sensors=[_Sensor()],
        controller=controller,
        max_steps=2,
    )

    orch.run()

    # The second tick's measurement carries the first tick's state: sampled 0, actuated to 1, stepped to 10.
    assert controller.meas.temperature == 10
