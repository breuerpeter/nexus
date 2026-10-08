"""The contracts the loop drives take no ambient sample: physics, the actuator and each sensor read the
state and the clock, and nothing else. Warp-free stubs, as in test_orchestrator_stop.py.
"""

from dataclasses import dataclass

import pytest

from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.core.orchestrator import Orchestrator
from nexus_sim._src.core.schema import Controls, SimTime
from nexus_sim._src.core.signals import Signal

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


@dataclass
class _Reading:
    """The state's one number, as the sensor sampled it: a host signal's value."""

    q: int


class _Sensor:
    """Copies the state into the signal `q`, so what the controller reads shows the sensor ran."""

    def __init__(self):
        self.out = Signal("q", _Reading)

    def sample(self, state, t):
        self.out.write(_Reading(state["q"]))

    def stages(self):
        return [Stage("sample", "host", lambda tick: self.sample(tick.state, tick.t), writes=(self.out,))]


class _Controller:
    """Answers the seed pass at once and keeps the last reading of the signal `q` its exchange read."""

    def __init__(self):
        self.q = Signal("q", _Reading)
        self.last = None

    def connect(self):
        pass

    def exchange(self, t, timeout):
        self.last = self.q.read()
        return Controls(command=[0.0, 0.0, 0.0, 0.0])

    def close(self):
        pass

    def stages(self):
        def exchange(tick):
            return self.exchange(tick.t, None) is not None

        return [Stage("exchange", "host", exchange, reads=(self.q,))]


def test_the_loop_runs_a_tick_on_seams_that_take_no_env():
    """The loop runs a tick on contracts that take no `env`: `Physics.step(state, dt)`,
    `Actuator.forces(controls, state)` and `Sensor.sample(state, t)`. A tick's exchange follows its step, so
    the second tick's sample carries the state after two ticks.
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

    # Tick one actuates to 1 and steps to 10; tick two to 11 and 110, which the sensor samples.
    assert controller.last == _Reading(110)
