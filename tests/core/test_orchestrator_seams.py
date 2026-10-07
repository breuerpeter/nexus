"""The seams the loop drives take no ambient sample: physics, the actuator and each sensor read the
state and the clock, and nothing else. Warp-free stubs, as in test_orchestrator_stop.py.
"""

import pytest

from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.core.orchestrator import Orchestrator
from nexus_sim._src.core.schema import Controls, SimTime

pytestmark = pytest.mark.usefixtures("warp_cpu")  # Python stand-ins run stage by stage, never as a graph


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

    def stages(self):
        return [
            Stage("clear", "device", lambda tick: self.clear_forces(tick.state)),
            Stage("step", "device", lambda tick: self.step(tick.state, tick.dt)),
        ]


class _Actuator:
    def forces(self, controls, state):
        state["q"] += 1

    def stages(self):
        return [Stage("forces", "device", lambda tick: self.forces(None, tick.state))]


class _Sensor:
    """Copies the state into the measurement, so what the controller receives shows the sensor ran."""

    def sample(self, state, t, out):
        out.temperature = state["q"]

    def stages(self):
        return [Stage("sample", "device", lambda tick: self.sample(tick.state, tick.t, tick.meas))]


class _Controller:
    """Answers the seed pass at once and keeps the last measurement it received: the read and
    exchange host stages over ``exchange``.
    """

    def __init__(self):
        self.meas = None

    def connect(self):
        pass

    def exchange(self, meas, t, timeout):
        self.meas = meas
        return Controls(command=[0.0, 0.0, 0.0, 0.0])

    def close(self):
        pass

    def stages(self):
        def exchange(tick):
            return self.exchange(tick.meas, tick.t, None) is not None

        return [Stage("read", "host", lambda tick: None), Stage("exchange", "host", exchange)]


def test_the_loop_runs_a_tick_on_seams_that_take_no_env():
    """The loop runs a tick on seams that take no `env`: `Physics.step(state, dt)`,
    `Actuator.forces(controls, state)` and `Sensor.sample(state, t, out)`. A tick's exchange follows
    its step, so the second tick's measurement carries the state after two ticks.
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

    # Sampled 0 at the seed pass; tick one actuates to 1 and steps to 10; tick two to 11 and 110.
    assert controller.meas.temperature == 110
