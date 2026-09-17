"""Orchestrator hosting hooks: step() drives the run; stop() ends the loop and closes the controller."""

import pytest

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


class _Env:
    def sample(self, pos, t):
        return None


class _Physics:
    def reset(self):
        return {"q": 0}

    def clear_forces(self, state):
        pass

    def step(self, state, env, dt):
        return state


class _Actuator:
    def forces(self, controls, state, env):
        pass


class _Controller:
    """Lockstep stand-in: returns controls immediately, so preroll succeeds at once."""

    def __init__(self):
        self.closed = False

    def connect(self):
        pass

    def exchange(self, meas, t, timeout):
        return Controls(command=[0.0, 0.0, 0.0, 0.0])

    def close(self):
        self.closed = True


def _orch(**kw):
    return Orchestrator(
        clock=_Clock(),
        environment=_Env(),
        physics=_Physics(),
        actuator=_Actuator(),
        sensors=[],
        controller=_Controller(),
        **kw,
    )


def test_step_drives_the_loop_and_stop_ends_it():
    """The whole hosting contract, with no thread in it: step() advances a tick on the calling
    thread, stop() ends the loop, and the generator's finally closes the controller on the way out.
    """
    orch = _orch()

    assert orch.step() is True, "the first step runs setup (preroll establishes lockstep) and one tick"
    assert orch.step() is True
    orch.stop()
    assert orch.step() is False, "stop() must end the steady loop"
    assert orch.controller.closed, "controller.close() runs in finally"


def test_a_controller_that_raises_on_close_still_finalizes_the_recording():
    """The recording is what a failed run is *for*. A controller whose teardown raises, as the PX4 one
    that stops a docker container in there can, must not cost the run `_close_logs`, which flushes each
    loggable's accumulated emission and closes the sink.
    """
    closed_logs = []

    def boom():
        raise RuntimeError("docker daemon went away")

    orch = _orch(max_steps=1)
    orch.controller.close = boom
    orch._close_logs = lambda: closed_logs.append(True)

    with pytest.raises(RuntimeError, match="docker daemon went away"):
        orch.run()  # the failure still surfaces: re-raised, not swallowed

    assert closed_logs == [True], "the recording teardown was skipped by the controller failure"
