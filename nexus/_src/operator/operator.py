"""The wall-clock wait a script uses on the PX4 offboard client's telemetry."""

from __future__ import annotations

import time
from collections.abc import Callable


def wait_until(predicate: Callable[[], bool], timeout: float, poll: float = 0.05) -> None:
    """Block until ``predicate()`` is true, or raise ``TimeoutError``.

    A wall-clock poll, so the predicate's source must refresh itself: it suits operator telemetry,
    say ``lambda: not operator.is_armed()``, which ``OffboardClient``'s pump thread keeps fresh. A
    sim-state predicate needs ``Sim.wait_until``, which steps the sim; nothing else advances it.

    Args:
        predicate: Zero-arg callable polled until it returns true.
        timeout: Wall-clock seconds to wait before giving up.
        poll: Sleep interval in wall-clock seconds between polls.

    Raises:
        TimeoutError: ``predicate()`` didn't become true within ``timeout`` seconds.
    """
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return
        if time.monotonic() >= deadline:
            break
        time.sleep(poll)
    raise TimeoutError(f"condition not met within {timeout}s")
