"""The estimator's place in the tick: its stage runs after the sensors' and before the controller's, so the
controller reads an estimate made from that tick's sensor values. Stand-in components at the loop's roles; each
CUDA test scopes the device it uses, so the default device is the same after it.
"""

import numpy as np
import pytest

pytest.importorskip("warp")

import warp as wp

from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.core.orchestrator import Orchestrator
from nexus_sim._src.core.schema import PoseTwist, SimTime
from nexus_sim._src.core.signals import DeviceType, Signal

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]


class Count(DeviceType):
    """A signal type this test defines: one 32-bit count."""

    dtype = wp.int32


@wp.kernel
def _count(n: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
    n[0] = n[0] + 1
    out[0] = n[0]


@wp.kernel
def _estimate(count: wp.array(dtype=wp.int32), estimate: wp.array2d(dtype=float)):
    estimate[0, 0] = float(count[0])


@wp.kernel
def _copy_first(estimate: wp.array2d(dtype=float), seen: wp.array(dtype=float)):
    seen[0] = estimate[0, 0]


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
    """The physics state: one body at rest at the origin."""

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
    stage stays out of the warm pass, so the count is the tick's.
    """

    def __init__(self):
        self.count = Signal("count", Count, shape=(1,))
        self._n = wp.zeros(1, dtype=wp.int32)

    def stages(self):
        return [Stage("count", "device", self._run, warm=False, writes=(self.count,))]

    def _run(self, tick):
        wp.launch(_count, dim=1, inputs=[self._n], outputs=[self.count.buffer])


class _CountingEstimator:
    """An estimator whose device stage writes the count it reads into the first value of its estimate."""

    def __init__(self):
        self.count = Signal("count", Count, shape=(1,))
        self.estimate = Signal("estimate", PoseTwist, shape=(1, 13))

    def stages(self):
        return [Stage("estimate", "device", self._run, reads=(self.count,), writes=(self.estimate,))]

    def _run(self, tick):
        wp.launch(_estimate, dim=1, inputs=[self.count.buffer], outputs=[self.estimate.buffer])


class _EstimateReader:
    """A controller whose device stage copies the first value of the estimate into a buffer of its own, `seen`."""

    def __init__(self):
        self.estimate = Signal("estimate", PoseTwist, shape=(1, 13))
        self.seen = wp.zeros(1, dtype=float)

    def connect(self):
        pass

    def close(self):
        pass

    def stages(self):
        return [Stage("read", "device", self._read, reads=(self.estimate,))]

    def _read(self, tick):
        wp.launch(_copy_first, dim=1, inputs=[self.estimate.buffer], outputs=[self.seen])


@pytest.mark.parametrize("device", DEVICES)
def test_the_controller_reads_an_estimate_made_from_that_ticks_sensor_values(device):
    """The estimator's stage runs after the sensors' and before the controller's, so the controller reads an
    estimate made from that tick's sensor values.

    Given a stand-in sensor whose device stage writes its tick count to a signal, a stand-in estimator that
    writes that count into its estimate, and a stand-in controller whose device stage reads the estimate, when
    the run steps 3 ticks, eagerly on the CPU device and captured on a CUDA device, then on each tick the
    controller reads the count the sensor wrote on that tick.
    """
    with wp.ScopedDevice(device):
        controller = _EstimateReader()
        orch = Orchestrator(
            clock=_Clock(),
            physics=_Physics(),
            sensors=[_Counter()],
            estimator=_CountingEstimator(),
            controller=controller,
        )
        seen = []
        for _ in range(3):
            orch.step()
            seen.append(float(controller.seen.numpy()[0]))
        orch.close()
    assert seen == [1.0, 2.0, 3.0]
