"""Clock: fixed-step sim time + optional real-time scaling, and its mirror on the device.

The sim time is a whole number of nanoseconds, the tick count times the tick, as gz-sim keeps it, so a time
that falls on a tick compares exactly. A stage and a row read it in seconds, the nanoseconds over 1e9.
"""

from __future__ import annotations

import time

import warp as wp

from .interfaces import Stage
from .schema import SimTime
from .signals import Signal


class Clock:
    """Owns the tick count and the rtf wall-clock throttle.

    The loop advances it at the start of each tick, then throttles to ``dt / rtf`` after. When PX4's blocking
    lockstep paces the loop, ``rtf`` stays at 0, which means no throttle, and PX4 drives the cadence.

    Args:
        dt: The control tick, seconds, held as a whole number of nanoseconds.
        rtf: The real-time factor to pace to; 0 runs as fast as it can.
    """

    def __init__(self, dt: float, rtf: float = 0.0):
        self.dt = dt
        self.dt_ns = round(dt * 1e9)
        self.rtf = rtf
        self._t = SimTime(0.0, 0)
        self._deadline: float | None = None  # cumulative wall-clock pace target

    def now(self) -> SimTime:
        return self._t

    def advance(self) -> SimTime:
        """Count one tick, and return the new time."""
        step = self._t.step_index + 1
        self._t = SimTime(step * self.dt_ns / 1e9, step)
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
def _add_tick(dt_ns: wp.int64, ns: wp.array(dtype=wp.int64), seconds: wp.array(dtype=wp.float64)):
    ns[0] = ns[0] + dt_ns
    seconds[0] = wp.float64(ns[0]) / wp.float64(1.0e9)


class DeviceClock:
    """The tick's sim time on the device: the signal ``time``, in seconds, which a device stage reads to stamp
    what it writes, and ``time_ns``, the same time in nanoseconds, which a sensor's due time compares with. Both
    hold inside a captured graph too.

    Its one device stage, ``clock``, adds one control tick at the start of each tick, as :class:`Clock` does on
    the host, so the two hold the same time. The loop sets the start time before the first tick.

    Args:
        dt: The control tick, seconds, held as a whole number of nanoseconds.
    """

    name = "clock"

    def __init__(self, dt: float):
        self.dt_ns = round(dt * 1e9)
        self.time = Signal("time", wp.float64, shape=(1,))
        self.time_ns = Signal("time_ns", wp.int64, shape=(1,))

    def stages(self) -> list[Stage]:
        """The one device stage, ``clock``, which advances the time by one control tick."""
        return [Stage("clock", "device", lambda tick: self.advance(), writes=(self.time, self.time_ns))]

    def set(self, t: SimTime) -> None:
        """Set the time to ``t``, from the host, before the first tick."""
        self.time_ns.write([t.step_index * self.dt_ns])
        self.time.write([t.sim_time])

    def advance(self) -> None:
        """Add one control tick to the time, on the device."""
        wp.launch(_add_tick, dim=1, inputs=(wp.int64(self.dt_ns),), outputs=(self.time_ns.buffer, self.time.buffer))
