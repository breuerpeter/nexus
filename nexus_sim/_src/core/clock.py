"""Clock: fixed-step sim time + optional real-time scaling, and its mirror on the device."""

from __future__ import annotations

import time

import warp as wp

from .interfaces import Stage
from .schema import SimTime
from .signals import Signal


class Clock:
    """Owns sim_time + step index and the rtf wall-clock throttle.

    sim_time += dt before the step,
    then throttle to ``sim_dt / rtf`` after. When PX4's blocking lockstep paces the
    loop, ``rtf`` stays at 0, which means no throttle, and PX4 drives the cadence.
    """

    def __init__(self, dt: float, rtf: float = 0.0):
        self.dt = dt
        self.rtf = rtf
        self._t = SimTime(0.0, 0)
        self._deadline: float | None = None  # cumulative wall-clock pace target

    def now(self) -> SimTime:
        return self._t

    def advance(self) -> SimTime:
        """Increment by dt *before* the step, and return the new time."""
        self._t = SimTime(self._t.sim_time + self.dt, self._t.step_index + 1)
        return self._t

    def throttle(self) -> None:
        """Apply the rtf wall-clock throttle, a no-op when rtf == 0.

        Paces against a cumulative deadline, not per-tick: the following fast ticks amortize a
        slow tick, say a 15 ms render inside a 4 ms budget, so ``rtf=1`` delivers a true 1.0x
        whenever the average tick fits the budget. The old per-tick reset could never repay an
        overrun and drifted to ~0.7x under render load. A cap on the repayable debt makes a long
        stall, such as a cesium tile-ingestion spike, resume pacing instead of triggering a burst.
        """
        if self.rtf <= 0:
            return
        now = time.time()
        if self._deadline is None:
            self._deadline = now
        self._deadline += self.dt / self.rtf
        if self._deadline > now:
            time.sleep(self._deadline - now)
        elif now - self._deadline > 0.25:
            self._deadline = now - 0.25  # cap the catch-up debt: pace forward, don't sprint


@wp.kernel
def _add_tick(dt: wp.float64, now: wp.array(dtype=wp.float64)):
    now[0] = now[0] + dt


class DeviceClock:
    """The tick's sim time on the device: the signal ``time``, which a device stage reads to stamp what it
    writes, inside a captured graph too.

    Its one device stage, ``clock``, adds one control tick at the start of each tick, as :class:`Clock` does on
    the host, so the two hold the same time. The loop writes the start time before the first tick.

    Args:
        dt: The control tick, seconds.
    """

    name = "clock"

    def __init__(self, dt: float):
        self.dt = float(dt)
        self.time = Signal("time", wp.float64, shape=(1,))

    def stages(self) -> list[Stage]:
        """The one device stage, ``clock``, which advances the time by one control tick."""
        return [Stage("clock", "device", lambda tick: self.advance(), writes=(self.time,))]

    def advance(self) -> None:
        """Add one control tick to the time, on the device."""
        wp.launch(_add_tick, dim=1, inputs=(self.dt,), outputs=(self.time.buffer,))
