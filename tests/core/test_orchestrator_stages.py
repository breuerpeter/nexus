"""The loop runs each component's work as device and host stages: every component states its stages,
each maximal run of device stages replays as one CUDA graph, and the host stages run between replays.
Stand-in components at the loop's seams; the CUDA tests skip without a device and scope the one they
use, so the default device is the same after them.
"""

import logging

import pytest

pytest.importorskip("warp")

import warp as wp

from nexus._src.core.interfaces import Stage
from nexus._src.core.orchestrator import Orchestrator
from nexus._src.core.schema import Controls, SimTime


@wp.kernel
def _bump(a: wp.array(dtype=wp.int32)):
    a[0] = a[0] + 1


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
    """Device physics: ``step`` bumps a device counter, so a test reads how many physics steps ran."""

    capturable = True

    def __init__(self):
        self.steps = None

    def reset(self):
        self.steps = wp.zeros(1, dtype=wp.int32)
        return {"q": 0}

    def clear_forces(self, state):
        pass

    def step(self, state, dt):
        wp.launch(_bump, dim=1, inputs=[self.steps])
        return state

    def stages(self):
        return [
            Stage("clear", "device", lambda tick: self.clear_forces(tick.state)),
            Stage("step", "device", lambda tick: self.step(tick.state, tick.dt)),
        ]


class _Actuator:
    capturable = True

    def forces(self, controls, state):
        pass

    def forces_wp(self, state):
        pass

    def write_controls(self, controls):
        pass

    def stages(self):
        return [Stage("forces", "device", lambda tick: self.forces(tick.controls, tick.state))]


class _GraphSensor:
    """An in-graph sensor: one device stage that writes nothing a stand-in needs."""

    capturable = True

    def sample(self, state, t, meas):
        pass

    def sample_wp(self, state, t):
        pass

    def read(self, meas):
        pass

    def stages(self):
        return [Stage("sample", "device", lambda tick: None)]


class _HostSensor:
    """A sensor whose work is a host stage, as an RTX sensor's frame exchange with the Kit peer is. It
    keeps, per sample, whether the device was under capture at the time.
    """

    host_rate = True

    def __init__(self):
        self.captured = []

    def sample(self, state, t, meas):
        self.captured.append(wp.get_device().is_capturing)

    def stages(self):
        return [Stage("sample", "host", lambda tick: self.sample(tick.state, tick.t, tick.meas))]


class _PeerController:
    """A controller with a peer, the PX4 shape: a ``read`` and an ``exchange`` host stage, plus one
    device stage that bumps its own buffer, which exists from construction because the loop captures
    before a peer connects. ``answers`` says which exchanges the peer answers; ``attached`` says when
    the peer dialed in. It keeps every exchange and device-stage call.
    """

    host_boundary = True

    def __init__(self, *, attached=True, answers=lambda n: True):
        self.attached = attached
        self._answers = answers
        self.calls = 0
        self.device_calls = 0
        self.act = wp.zeros(1, dtype=wp.int32)
        self.closed = False

    def connect(self):
        pass

    def exchange(self, meas, t, timeout):
        self.calls += 1
        if not self._answers(self.calls):
            return None
        return Controls(command=[0.0, 0.0, 0.0, 0.0])

    def close(self):
        self.closed = True

    def _act(self, tick):
        self.device_calls += 1
        wp.launch(_bump, dim=1, inputs=[self.act])

    def stages(self):
        def exchange(tick):
            tick.controls = self.exchange(tick.meas, tick.t, None)
            return tick.controls is not None

        return [
            Stage("read", "host", lambda tick: None),
            Stage("exchange", "host", exchange),
            Stage("act", "device", self._act, warm=False),
        ]


class _DeviceController:
    """A controller whose stages are all device stages, the Proportional Integral Derivative (PID) shape. It keeps every ``exchange`` call."""

    capturable = True

    def __init__(self):
        self.exchange_calls = 0
        self.act = None

    def connect(self):
        self.act = wp.zeros(1, dtype=wp.int32)

    def exchange(self, meas, t, timeout):
        self.exchange_calls += 1
        return Controls(command=[0.0, 0.0, 0.0, 0.0])

    def close(self):
        pass

    def stages(self):
        return [Stage("act", "device", lambda tick: wp.launch(_bump, dim=1, inputs=[self.act]))]


class _Stageless:
    """A controller that states no stages."""

    def connect(self):
        pass

    def close(self):
        pass


class _EmptyStages:
    """A controller whose stage list is empty."""

    def connect(self):
        pass

    def close(self):
        pass

    def stages(self):
        return []


class _UnknownKind:
    """A controller whose one stage misspells its kind."""

    def connect(self):
        pass

    def close(self):
        pass

    def stages(self):
        return [Stage("exchange", "hots", lambda tick: None)]


class _Logger:
    """The recording sink, the shape the loop calls: it keeps whether the run closed it."""

    log_interval = 0.0

    def __init__(self):
        self.closed = False

    def set_time(self, t):
        pass

    def close(self):
        self.closed = True


def _orch(controller, *, sensors=(), physics=None, **kw):
    return Orchestrator(
        clock=_Clock(),
        physics=physics or _Physics(),
        actuator=_Actuator(),
        sensors=[_GraphSensor(), *sensors],
        controller=controller,
        **kw,
    )


def _messages(caplog):
    return [r.getMessage() for r in caplog.records]


def _cuda():
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    return wp.ScopedDevice("cuda:0")


def test_a_controllers_device_stages_replay_in_the_graph_and_its_host_stages_run_between():
    """A controller written against the stage contract flies: its device stages replay inside the graph
    and its host stages run once per tick between replays. Over two steady ticks the host stage ran
    twice on the host, the device stage's buffer advanced twice, and its Python ran zero times: the
    graph replayed it.
    """
    with _cuda():
        controller = _PeerController()
        orch = _orch(controller)
        orch.step()
        calls, device_calls, act = controller.calls, controller.device_calls, int(controller.act.numpy()[0])
        orch.step()
        orch.step()
        wp.synchronize()
        delta = (
            controller.calls - calls,
            int(controller.act.numpy()[0]) - act,
            controller.device_calls - device_calls,
        )
        orch.close()
    assert delta == (2, 2, 0)


@pytest.mark.parametrize(
    "controller", [_Stageless(), _EmptyStages(), _UnknownKind()], ids=["no stages", "empty stages", "unknown kind"]
)
def test_a_component_with_no_stages_fails_the_build_naming_it(controller):
    """A component with no stages, or a stage of a kind the loop doesn't know, fails the build with an
    error naming the component; nothing falls back to eager in silence. On CPU the same.
    """
    with wp.ScopedDevice("cpu"), pytest.raises(ValueError, match=type(controller).__name__):
        _orch(controller).step()


def test_a_run_logs_its_stage_plan_with_its_host_stages(caplog):
    """A run logs its stage plan once at start: each captured segment and the host stages between them.
    The PX4 shape: one line naming one segment and the ``read`` and ``exchange`` host stages.
    """
    with _cuda(), caplog.at_level(logging.INFO, logger="nexus"):
        orch = _orch(_PeerController())
        orch.step()
        orch.close()
    plans = [m for m in _messages(caplog) if m.startswith("stage plan:")]
    assert len(plans) == 1 and plans[0].count("graph(") == 1 and "read" in plans[0] and "exchange" in plans[0]


def test_a_run_logs_its_stage_plan_as_one_segment_for_device_stages_only(caplog):
    """A run logs its stage plan once at start. The Proportional Integral Derivative (PID) shape: one
    segment and no host stage.
    """
    with _cuda(), caplog.at_level(logging.INFO, logger="nexus"):
        orch = _orch(_DeviceController())
        orch.step()
        orch.close()
    plans = [m for m in _messages(caplog) if m.startswith("stage plan:")]
    assert len(plans) == 1 and plans[0].count("graph(") == 1 and "host(" not in plans[0]


def test_a_controller_with_only_device_stages_gets_no_preroll_and_no_exchange_call():
    """A controller whose stages are all device stages gets no preroll and no exchange call: after three
    ticks ``exchange`` was never called and the clock advanced once per tick from the first tick.
    """
    with wp.ScopedDevice("cpu"):
        controller = _DeviceController()
        orch = _orch(controller)
        orch.step()
        orch.step()
        orch.step()
        seen = (controller.exchange_calls, orch.clock.now().step_index)
        orch.close()
    assert seen == (0, 3)


def test_a_peer_that_never_attaches_ends_the_setup_with_a_timeout_error(caplog):
    """A peer that never attaches ends the setup with a timeout error that names the wait: with a preroll
    timeout of 0.2 s the run ends on its first step, the log names the 0.2 s, and the run finalizes
    the recording.
    """
    with wp.ScopedDevice("cpu"), caplog.at_level(logging.INFO, logger="nexus"):
        sink = _Logger()
        orch = _orch(_PeerController(attached=False, answers=lambda n: False), logger=sink, preroll_timeout=0.2)
        ran = orch.step()
    assert (ran, any("0.2" in m for m in _messages(caplog)), sink.closed) == (False, True, True)


def test_a_peer_that_stops_answering_mid_flight_ends_the_run_normally(caplog):
    """A peer that stops answering mid-flight ends the run as its normal end, and the run finalizes
    the recording: the peer answers the settled exchange and nine ticks, so ``step()`` returns ``False`` on
    the tenth, the run logs the disconnect, and the run closes the sink.
    """
    with wp.ScopedDevice("cpu"), caplog.at_level(logging.INFO, logger="nexus"):
        sink = _Logger()
        orch = _orch(_PeerController(answers=lambda n: n <= 10), logger=sink)
        steps = [orch.step() for _ in range(10)]
    assert (steps, any("disconnected" in m for m in _messages(caplog)), sink.closed) == (
        [True] * 9 + [False],
        True,
        True,
    )


def test_physics_substeps_run_that_many_physics_steps_per_tick_when_captured():
    """``physics_substeps`` greater than 1 runs that many physics steps per tick on the captured path, as eager
    does: with two substeps, five ticks add ten physics steps.
    """
    with _cuda():
        physics = _Physics()
        orch = _orch(_DeviceController(), physics=physics, physics_substeps=2)
        for _ in range(5):
            orch.step()
        wp.synchronize()
        at_five = int(physics.steps.numpy()[0])
        for _ in range(5):
            orch.step()
        wp.synchronize()
        delta = int(physics.steps.numpy()[0]) - at_five
        orch.close()
    assert delta == 10


def test_a_host_stage_sensor_samples_once_per_tick_outside_the_graph():
    """A sensor whose work is a host stage samples once per tick at the host seam, outside the graph:
    over three ticks it sampled three times, never under capture.
    """
    with _cuda():
        rtx = _HostSensor()
        orch = _orch(_PeerController(), sensors=[rtx])
        orch.step()
        orch.step()
        orch.step()
        orch.close()
    assert rtx.captured == [False, False, False]
