"""The stops that reach a run from outside: SIGTERM and SIGHUP as Ctrl-C, and a stop held through a write.

The tests send the signals to their own process, with the handlers under test installed, so none
reaches the default action.
"""

import os
import signal
import threading
import time

import pytest

from nexus_sim._src.core.interrupts import hold_interrupts, stop_signals


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP], ids=["SIGTERM", "SIGHUP"])
def test_a_stop_signal_raises_keyboard_interrupt_inside_the_block(sig):
    """SIGTERM and SIGHUP end a run as Ctrl-C does: inside the block each raises `KeyboardInterrupt`."""
    with pytest.raises(KeyboardInterrupt), stop_signals():
        os.kill(os.getpid(), sig)
        time.sleep(0.1)  # the handler runs on the way back from the sleep


def test_the_stop_handlers_go_back_after_the_block():
    """After the block the process keeps its own handling of SIGTERM and SIGHUP."""
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    with stop_signals():
        pass
    assert {sig: signal.getsignal(sig) for sig in before} == before


def test_a_held_interrupt_lets_the_block_finish_and_raises_once_after_it():
    """A Ctrl-C that lands inside the held block waits: the block runs to its end, then one
    `KeyboardInterrupt` goes up.
    """
    finished = []
    with pytest.raises(KeyboardInterrupt):
        with hold_interrupts():
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.1)
            finished.append(True)
    assert finished == [True]


def test_a_block_with_no_interrupt_raises_nothing_and_puts_the_handlers_back():
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    with hold_interrupts():
        pass
    assert {sig: signal.getsignal(sig) for sig in before} == before


def test_off_the_main_thread_both_are_no_ops():
    """Python delivers signals to the main thread alone, so a run stepped from another thread leaves the
    process's handling as it stands.
    """
    before = signal.getsignal(signal.SIGTERM)
    seen = []

    def run():
        with stop_signals(), hold_interrupts():
            seen.append(signal.getsignal(signal.SIGTERM))

    t = threading.Thread(target=run)
    t.start()
    t.join()
    assert seen == [before]
