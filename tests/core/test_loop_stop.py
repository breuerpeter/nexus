"""Loop hosting hooks: step() drives the run; stop() ends the loop and closes the controller."""

import pytest

from nexus_uav._src.core.errors import RunError
from nexus_uav._src.core.interfaces import Component, Stage
from nexus_uav._src.core.loop import Loop
from nexus_uav._src.core.schema import SimTime

pytestmark = pytest.mark.usefixtures("warp_cpu")  # Python stand-ins run stage by stage, never as a graph


class _Clock:
    dt = 0.004

    def __init__(self):
        self._t = SimTime()

    def now(self):
        return self._t

    def advance(self):
        self._t = SimTime(self._t.time_ns + round(self.dt * 1e9), self._t.step_index + 1)
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

    def stages(self):
        return [
            Stage("clear", "device", lambda tick: self.clear_forces(tick.state)),
            Stage("step", "device", lambda tick: self.step(tick.state, tick.dt)),
            Stage("state", "device", lambda tick: None),
        ]


class _Actuator(Component):
    def forces(self, controls, state):
        pass

    def stages(self):
        return [Stage("forces", "device", lambda tick: self.forces(None, tick.state))]


class _Controller(Component):
    """Lockstep stand-in: returns controls immediately, so preroll succeeds at once. Its work is the
    PX4 shape, a ``truth`` and an ``exchange`` host stage.
    """

    def __init__(self):
        self.closed = False

    def connect(self, links):
        pass

    def exchange(self, t, timeout):
        return [0.0, 0.0, 0.0, 0.0]

    def close(self):
        self.closed = True

    def stages(self):
        def exchange(tick):
            return self.exchange(tick.t, None) is not None

        return [Stage("truth", "host", lambda tick: None), Stage("exchange", "host", exchange)]


def _orch(sensors=(), **kw):
    return Loop(
        clock=_Clock(),
        physics=_Physics(),
        actuator=_Actuator(),
        sensors=list(sensors),
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


def test_a_physics_reset_that_fails_still_stops_the_peers():
    """The peers start at build, before the run: a reset that raises, a Warp kernel that fails to compile, must
    not leave one running, the render stage or a container.
    """

    def boom():
        raise RuntimeError("CUDA kernel build failed")

    peer = _Peer()
    orch = _orch(peers=[peer])
    orch.physics.reset = boom

    with pytest.raises(RuntimeError, match="CUDA kernel build failed"):
        orch.run()

    assert peer.stopped == 1


class _DyingSensor(Component):
    """A sensor whose work is a host stage and whose renderer fails mid-flight, as an RTX sensor's can."""

    host_rate = True

    def sample(self, state, t):
        raise RunError("the render stage failed while this run rendered a frame")

    def stages(self):
        return [Stage("sample", "host", lambda tick: self.sample(tick.state, tick.t))]


def test_a_sensor_whose_renderer_fails_mid_flight_ends_the_run_with_its_error_after_the_teardown():
    """Not the autopilot's disconnect, which ends a run normally: the error surfaces after the teardown."""
    peer = _Peer()
    orch = _orch(sensors=[_DyingSensor()], peers=[peer])

    with pytest.raises(RunError, match="render stage"):
        orch.run()

    assert (peer.stopped, orch.controller.closed) == (1, True)


class _Peer:
    """A peer the build started, on the peer contract: the loop only ever stops it."""

    def __init__(self):
        self.stopped = 0

    def start(self):
        raise AssertionError("the loop starts no peer; the build does")

    def stop(self):
        self.stopped += 1

    def alive(self):
        return self.stopped == 0


def test_a_run_that_ends_stops_the_peers_the_build_started():
    peer = _Peer()
    orch = _orch(max_steps=2, peers=[peer])

    orch.run()

    assert peer.stopped == 1


def test_closing_a_run_that_never_stepped_stops_its_peers():
    """A `Sim` entered and exited without a step still stops the PX4 container that started at build."""
    peer = _Peer()
    orch = _orch(peers=[peer])

    orch.close()

    assert peer.stopped == 1


def test_a_peer_whose_stop_fails_does_not_cost_the_recording():
    """A docker daemon that went away must not skip `_close_logs`, which flushes the recording."""
    closed_logs = []

    class Boom(_Peer):
        def stop(self):
            raise RuntimeError("docker daemon went away")

    orch = _orch(max_steps=1, peers=[Boom()])
    orch._close_logs = lambda: closed_logs.append(True)

    orch.run()

    assert closed_logs == [True]
