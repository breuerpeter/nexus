"""The tick's sim time on the device: the signal `time`, which the loop's clock stage advances at the start of
each tick in step with the host clock, so a device stage reads the tick's time, inside a captured graph too.
Stand-in components at the loop's roles; each CUDA test scopes the device it uses, so the default device is
the same after it.
"""

import pytest

pytest.importorskip("warp")

import warp as wp

from nexus_sim._src.core.clock import Clock
from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.core.orchestrator import Orchestrator
from nexus_sim._src.core.signals import Signal

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]


@wp.kernel
def _copy_time(time: wp.array(dtype=wp.float64), seen: wp.array(dtype=wp.float64)):
    seen[0] = time[0]


class _Physics:
    """Physics that moves nothing."""

    def reset(self):
        return None

    def stages(self):
        return [Stage("clear", "device", lambda tick: None), Stage("step", "device", lambda tick: None)]


class _Stamper:
    """A sensor whose device stage copies the signal `time` into a buffer of its own, `seen`."""

    def __init__(self):
        self.time = Signal("time", wp.float64, shape=(1,))
        self.seen = wp.zeros(1, dtype=wp.float64)

    def stages(self):
        return [Stage("stamp", "device", self._run, reads=(self.time,))]

    def _run(self, tick):
        wp.launch(_copy_time, dim=1, inputs=[self.time.buffer], outputs=[self.seen])


class _Keeper:
    """A controller whose host stage keeps the time the stamper copied, beside the tick's sim time."""

    def __init__(self, stamper: _Stamper):
        self._stamper = stamper
        self.kept = []

    def connect(self):
        pass

    def close(self):
        pass

    def stages(self):
        def keep(tick):
            self.kept.append((tick.t.sim_time, float(self._stamper.seen.numpy()[0])))
            return True

        return [Stage("keep", "host", keep)]


@pytest.mark.parametrize("device", DEVICES)
def test_a_device_stage_reads_the_ticks_sim_time_from_the_signal_time(device):
    """A device stage reads the tick's sim time from the signal `time`.

    Given a stand-in sensor whose device stage copies the signal `time`, and a stand-in controller whose host
    stage keeps that copy, when the run steps 3 ticks of 4 ms, eagerly on the CPU device and captured on a
    CUDA device, then on each tick the copy holds that tick's sim time: 8, 12 and 16 ms, after the seed pass's
    4 ms.
    """
    with wp.ScopedDevice(device):
        stamper = _Stamper()
        keeper = _Keeper(stamper)
        orch = Orchestrator(clock=Clock(0.004), physics=_Physics(), sensors=[stamper], controller=keeper)
        for _ in range(3):
            orch.step()
        orch.close()

    assert keeper.kept[-3:] == [pytest.approx((t, t), abs=1e-12) for t in (0.008, 0.012, 0.016)]
