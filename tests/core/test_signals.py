"""Components declare typed signals, and the builder wires them once: each stage declares the signals it
reads and writes, the loop hands the writer and every reader one buffer before the capture, and a
declaration that doesn't fit stops the run before any stage runs. Stand-in components at the loop's
roles; each CUDA test scopes the device it uses, so the default device is the same after it.
"""

import numpy as np
import pytest

pytest.importorskip("warp")

import warp as wp

from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.core.orchestrator import Orchestrator
from nexus_sim._src.core.schema import ReferenceTrajectory, SimTime
from nexus_sim._src.core.signals import DeviceType, Signal
from nexus_sim._src.guidance import TrackingGuidance

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]


class Count(DeviceType):
    """A signal type this test defines and core doesn't name: one 32-bit count."""

    dtype = wp.int32


@wp.kernel
def _count(n: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
    n[0] = n[0] + 1
    out[0] = n[0]


@wp.kernel
def _copy(src: wp.array(dtype=wp.int32), dst: wp.array(dtype=wp.int32)):
    dst[0] = src[0]


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
    """The physics state: one body at rest at the origin, which a guidance reads."""

    def __init__(self):
        self.body_q = wp.array(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32), dtype=wp.transform)


class _Physics:
    """Physics that moves nothing."""

    def reset(self):
        return _State()

    def stages(self):
        return [Stage("clear", "device", lambda tick: None), Stage("step", "device", lambda tick: None)]


class _Counter:
    """A sensor whose device stage counts the run's ticks and writes the count to the signal `count`. Its
    stage stays out of the warm pass, so the count is the tick's. It keeps how often its stage ran.
    """

    def __init__(self, name: str = "counter"):
        self.name = name
        self.count = Signal("count", Count, shape=(1,))
        self.ran = 0
        self._n = wp.zeros(1, dtype=wp.int32)

    def stages(self):
        return [Stage("count", "device", self._run, warm=False, writes=(self.count,))]

    def _run(self, tick):
        self.ran += 1
        wp.launch(_count, dim=1, inputs=[self._n], outputs=[self.count.buffer])


class _DeviceReader:
    """A controller whose device stage copies the signal `count` into a buffer of its own, `seen`."""

    def __init__(self):
        self.count = Signal("count", Count, shape=(1,))
        self.seen = wp.zeros(1, dtype=wp.int32)

    def connect(self):
        pass

    def close(self):
        pass

    def stages(self):
        return [Stage("read", "device", self._read, reads=(self.count,))]

    def _read(self, tick):
        wp.launch(_copy, dim=1, inputs=[self.count.buffer], outputs=[self.seen])


class _HostReader:
    """A controller whose host stage reads the signal `count` with a copy. It keeps each value it read."""

    def __init__(self):
        self.count = Signal("count", Count, shape=(1,))
        self.seen = []

    def connect(self):
        pass

    def close(self):
        pass

    def stages(self):
        return [Stage("read", "host", self._read, reads=(self.count,))]

    def _read(self, tick):
        self.seen.append(int(self.count.buffer.numpy()[0]))
        return True


class _KeepingReader:
    """A controller whose host stage keeps each value of the signal `count` it reads, as read gave it."""

    def __init__(self):
        self.count = Signal("count", Count, shape=(1,))
        self.kept = []

    def connect(self):
        pass

    def close(self):
        pass

    def stages(self):
        def keep(tick):
            self.kept.append(self.count.read())
            return True

        return [Stage("keep", "host", keep, reads=(self.count,))]


class _Reference:
    """A planned reference, the shape a tracking guidance writes."""

    duration = 4.0

    def set_start(self, p0):
        pass

    def reference_path(self):
        return []


class _SteerOnDevice:
    """A controller whose device stage, `steer`, reads the planned reference, which is a host signal. It
    keeps how often its stage ran.
    """

    def __init__(self):
        self.setpoint = Signal("setpoint", ReferenceTrajectory)
        self.ran = 0

    def connect(self):
        pass

    def close(self):
        pass

    def stages(self):
        return [Stage("steer", "device", self._steer, reads=(self.setpoint,))]

    def _steer(self, tick):
        self.ran += 1


def _orch(sensors, controller, **kw):
    return Orchestrator(clock=_Clock(), physics=_Physics(), sensors=sensors, controller=controller, **kw)


@pytest.mark.parametrize("device", DEVICES)
def test_a_component_reads_a_value_another_writes_of_a_type_core_does_not_name(device):
    """A component reads a value another component writes, of a type core doesn't name, with no change to
    the loop.

    Given a stand-in sensor whose device stage writes its tick count to a signal of a type the test defines,
    and a stand-in controller whose device stage reads it, when the run steps 3 ticks, eagerly on the CPU
    device and captured on a CUDA device, then on each tick the controller reads the count the sensor wrote
    on that tick.
    """
    with wp.ScopedDevice(device):
        controller = _DeviceReader()
        orch = _orch([_Counter()], controller)
        seen = []
        for _ in range(3):
            orch.step()
            seen.append(int(controller.seen.numpy()[0]))
        orch.close()
    assert seen == [1, 2, 3]


@pytest.mark.parametrize("device", DEVICES)
def test_a_host_stage_reads_the_value_a_device_stage_wrote_earlier_in_the_same_tick(device):
    """A host stage reads the value a device stage wrote earlier in the same tick.

    Given a stand-in sensor whose device stage writes its tick count to a signal, and a stand-in controller
    whose host stage reads it, when the run steps 3 ticks, eagerly on the CPU device and captured on a CUDA
    device, then on each tick the controller reads the count the sensor wrote on that tick.
    """
    with wp.ScopedDevice(device):
        controller = _HostReader()
        orch = _orch([_Counter()], controller)
        seen = []
        for _ in range(3):
            orch.step()
            seen.append(controller.seen[-1])
        orch.close()
    assert seen == [1, 2, 3]


@pytest.mark.parametrize("device", DEVICES)
def test_a_host_stage_keeps_the_copy_it_read_after_a_later_write(device):
    """A host stage reads a device signal with a copy, so what it keeps doesn't change at a later write.

    Given a stand-in sensor whose device stage writes its tick count to a signal, and a stand-in controller
    whose host stage keeps each value it reads, when the run steps 3 ticks, eagerly on the CPU device and
    captured on a CUDA device, then the controller keeps 1, 2 and 3.
    """
    with wp.ScopedDevice(device):
        controller = _KeepingReader()
        orch = _orch([_Counter()], controller)
        for _ in range(3):
            orch.step()
        orch.close()
    assert [int(value[0]) for value in controller.kept[-3:]] == [1, 2, 3]


def test_a_device_stage_that_reads_a_host_signal_fails_the_run_naming_the_stage_and_the_signal():
    """A device stage that reads or writes a host signal fails the run before any stage runs, and the error
    names the stage and the signal.

    Given a `TrackingGuidance` and a stand-in controller whose device stage reads the `ReferenceTrajectory` it
    writes, when the run starts, then it fails before any stage runs, and the error names that stage and the
    setpoint.
    """
    with wp.ScopedDevice("cpu"):
        guidance = TrackingGuidance(planner=lambda waypoints: _Reference())
        guidance.set_mission([(1.0, 0.0, 2.0)])
        controller = _SteerOnDevice()
        orch = _orch([], controller, guidance=guidance)
        with pytest.raises(ValueError) as e:
            orch.step()
        orch.close()
    named = all(word in str(e.value) for word in ("steer", "setpoint"))
    assert (named, controller.ran, guidance.reference_started_at) == (True, 0, None)


def test_two_components_that_write_one_signal_a_third_reads_fail_the_run_naming_both_writers():
    """Two components that write one signal that a third reads fail the run before any stage runs, and the
    error names both writers.

    Given two stand-in sensors whose device stages write one signal and a stand-in controller that reads it,
    when the run starts, then it fails before any stage runs, and the error names both sensors.
    """
    with wp.ScopedDevice("cpu"):
        writers = [_Counter("first_counter"), _Counter("second_counter")]
        orch = _orch(writers, _DeviceReader())
        with pytest.raises(ValueError) as e:
            orch.step()
        orch.close()
    named = all(name in str(e.value) for name in ("first_counter", "second_counter"))
    assert (named, [writer.ran for writer in writers]) == (True, [0, 0])
