"""The tick's sim time on the device: the signal `time`, which the loop's clock stage advances at the start of
each tick in step with the host clock, so a device stage reads the tick's time, inside a captured graph too.
Stand-in components at the loop's roles; each CUDA test scopes the device it uses, so the default device is
the same after it.
"""

import pytest

pytest.importorskip("warp")

import warp as wp

from nexus_sim._src.core.clock import Clock
from nexus_sim._src.core.interfaces import Component, Stage
from nexus_sim._src.core.loop import Loop
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
        return [
            Stage("clear", "device", lambda tick: None),
            Stage("step", "device", lambda tick: None),
            Stage("state", "device", lambda tick: None),
        ]


class _Stamper(Component):
    """A sensor whose device stage copies the signal `time` into a buffer of its own, `seen`."""

    def __init__(self):
        self.time = Signal("time", wp.float64, shape=(1,))
        self.seen = wp.zeros(1, dtype=wp.float64)

    def stages(self):
        return [Stage("stamp", "device", self._run, reads=(self.time,))]

    def _run(self, tick):
        wp.launch(_copy_time, dim=1, inputs=[self.time.buffer], outputs=[self.seen])


class _Keeper(Component):
    """A controller whose host stage keeps the time the stamper copied, beside the tick's sim time."""

    def __init__(self, stamper: _Stamper):
        self._stamper = stamper
        self.kept = []

    def connect(self, links):
        pass

    def close(self):
        pass

    def stages(self):
        def keep(tick):
            self.kept.append((tick.t.sim_time, float(self._stamper.seen.numpy()[0])))
            return True

        return [Stage("keep", "host", keep)]


@pytest.mark.parametrize("device", DEVICES)
def test_the_sim_time_is_the_tick_count_times_the_tick_on_the_host_and_on_the_device(device):
    """The sim time is the tick count times the tick in nanoseconds, on the host and on the device.

    Given a run at a 4 ms tick whose sensor copies the device's `time` and whose controller keeps that copy
    beside the tick's sim time, when it steps 10 ticks, eagerly on the CPU device and captured on a CUDA device,
    then on tick k both equal `k * 4_000_000 / 1e9` exactly, the tenth `0.04`, not `0.04000000000000001`.
    """
    with wp.ScopedDevice(device):
        stamper = _Stamper()
        keeper = _Keeper(stamper)
        orch = Loop(clock=Clock(0.004), physics=_Physics(), sensors=[stamper], controller=keeper)
        for _ in range(10):
            orch.step()
        orch.close()

    ticks = (0.004, 0.008, 0.012, 0.016, 0.02, 0.024, 0.028, 0.032, 0.036, 0.04)
    assert keeper.kept[-10:] == [(t, t) for t in ticks]
