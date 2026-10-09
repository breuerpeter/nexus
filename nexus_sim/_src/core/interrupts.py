"""The stops a run sees from outside: SIGTERM and SIGHUP end a run as Ctrl-C does, and a stop that
lands while the recording is written waits until the file is closed.

Both are context managers over the process's signal handlers. Python delivers a signal to the main
thread alone and refuses a handler from any other, so off the main thread each is a no-op: a run
stepped from another thread keeps the process's own handling.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager

# The signals that end a run as Ctrl-C does: a `kill`, a container's stop, a closed terminal.
STOP_SIGNALS = (signal.SIGTERM, signal.SIGHUP)


def _on_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


def _restore(previous: dict) -> None:
    for sig, handler in previous.items():
        signal.signal(sig, signal.SIG_DFL if handler is None else handler)


@contextmanager
def stop_signals() -> Iterator[None]:
    """Make SIGTERM and SIGHUP raise ``KeyboardInterrupt``, as SIGINT does, for the span of the block,
    and put the handlers back after it. So a run ended by any of the three runs its teardown.
    """
    if not _on_main_thread():
        yield
        return
    previous = {sig: signal.signal(sig, signal.default_int_handler) for sig in STOP_SIGNALS}
    try:
        yield
    finally:
        _restore(previous)


@contextmanager
def hold_interrupts() -> Iterator[None]:
    """Hold SIGINT, SIGTERM and SIGHUP for the span of the block: a signal that lands inside is noted,
    the block runs to its end, and one ``KeyboardInterrupt`` is raised once it has. So a second Ctrl-C
    during the recording's write waits until the file is closed.
    """
    if not _on_main_thread():
        yield
        return
    pending: list[int] = []

    def note(signum, frame) -> None:
        pending.append(signum)

    previous = {sig: signal.signal(sig, note) for sig in (signal.SIGINT, *STOP_SIGNALS)}
    try:
        yield
    finally:
        _restore(previous)
    if pending:
        raise KeyboardInterrupt
