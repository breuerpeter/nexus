"""The operator helpers: wait_until polling semantics."""

import pytest

from nexus._src.operator import wait_until


def test_wait_until_returns_when_predicate_true():
    calls = {"n": 0}

    def pred():
        calls["n"] += 1
        return calls["n"] >= 3

    wait_until(pred, timeout=2.0, poll=0.001)
    assert calls["n"] >= 3


def test_wait_until_raises_on_timeout():
    with pytest.raises(TimeoutError):
        wait_until(lambda: False, timeout=0.1, poll=0.01)


def test_wait_until_returns_immediately_when_already_true_zero_timeout():
    # predicate already true + timeout=0 must return, not raise, since wait_until
    # must call the predicate at least once regardless of timeout.
    wait_until(lambda: True, timeout=0.0)
