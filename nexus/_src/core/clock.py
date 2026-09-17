"""Clock: fixed-step sim time + optional real-time scaling, per architecture.md §2."""

from __future__ import annotations

import time

from .schema import SimTime


class Clock:
    """Owns sim_time + step index and the rtf wall-clock throttle.

    Lifted from the bridge's ``Simulator.step``: sim_time += dt before the step,
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
        """Increment by dt, as the bridge did with sim_time += dt *before* the step, and return it."""
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
