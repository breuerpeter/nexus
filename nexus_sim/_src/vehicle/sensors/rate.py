"""The one rule every sensor decimates by, whether its stage runs on the device or on the host.

A sensor takes its first sample on the first tick its stage runs, and the k-th sample after the first is due at
the first's time plus k periods. The times are the clock's whole nanoseconds, and the rule compares the elapsed
time times the rate with k seconds, so it rounds no period: a due time that falls on a tick compares exactly, and a
rate that doesn't divide the tick rate neither quantises nor drifts. A rate of 0 makes every tick due, as does a
rate faster than the tick.
"""

from __future__ import annotations

import warp as wp


def checked_rate(rate: float, where: str) -> float:
    """The declared rate `rate`, in hertz, of the sensor declared at `where`.

    Raises:
        ValueError: The rate is negative, or not a number; the message names `where`. Sampling every tick would
            let a typo pass in silence.
    """
    rate = float(rate)
    if not rate >= 0.0:  # a rate that's not a number fails too
        raise ValueError(f"{where}: the sensor's nexus:rate is {rate:g} Hz; a rate is 0, a sample every tick, or more")
    return rate


@wp.func
def due(elapsed: wp.int64, taken: int, rate: wp.float64) -> bool:
    """Whether a sensor at `rate` hertz that has taken `taken` samples has its next one due `elapsed`
    nanoseconds after its first. A kernel calls it, and a host stage calls it from Python with Warp scalars.

    The products are exact in float64 below 2**53: for a rate of whole hertz, that is 10 h of sim time at 250 Hz.
    """
    if taken == 0 or rate <= wp.float64(0.0):
        return True
    return wp.float64(elapsed) * rate >= wp.float64(taken) * wp.float64(1.0e9)
