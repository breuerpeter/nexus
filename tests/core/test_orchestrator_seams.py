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
    def reset(self):
        return {"q": 0}

    def clear_forces(self, state):
        pass

    def step(self, state, dt):
        return state


class _Actuator:
    def forces(self, controls, state):
        pass


class _Sensor:
    """Fills one field, so the measurement the controller receives shows the sensor ran."""

    def sample(self, state, t, out):
        out.temperature = 21.0


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
        max_steps=1,
    )

    orch.run()

    assert controller.meas.temperature == 21.0
