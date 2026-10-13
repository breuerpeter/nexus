"""The instance a run holds on this machine, which numbers every address its peers use.

A run owns every address its peers use, so it picks one instance for every peer: the lowest one no other run
holds and that each peer it starts finds free. A peer's class states what makes an instance busy for it in an
optional ``instance_free(instance)``, such as a port of its own that another process holds. A run holds its
instance with a lock on a file of its own, taken without waiting, which the run releases when it stops its
peers. So two runs that start at once take two instances.

The lock folder keeps the name it had when PX4 alone claimed an instance, ``~/.cache/nexus/px4-instances/``, so
a run of an earlier release on this machine takes the same locks.
"""

from __future__ import annotations

import fcntl
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import IO

COUNT = 256  # the instances a machine holds


class InstanceClaim:
    """The run's hold on its instance: a lock file, which ``stop`` closes, freeing the instance.

    The run stops it with its peers, after them, so the instance stays held while any of them runs. It speaks the
    peer contract for that alone: it starts no process.
    """

    def __init__(self, instance: int, lock: IO) -> None:
        self.instance = instance
        self._lock: IO | None = lock

    def start(self) -> None:
        """Nothing: the claim holds its instance from the moment the run makes it."""

    def stop(self) -> None:
        """Release the instance. Idempotent."""
        if self._lock is not None:
            self._lock.close()  # closing the file releases its lock, and the instance is free again
            self._lock = None

    def alive(self) -> bool:
        """Whether the claim still holds its instance."""
        return self._lock is not None

    def artifacts(self) -> dict:
        """No file: the claim writes none."""
        return {}


def claim(peers: Iterable[type] = ()) -> InstanceClaim:
    """The lowest instance free on this machine for every class of `peers`, held by the claim returned.

    Raises:
        RuntimeError: Every one of 256 instances is in use.
    """
    checks: list[Callable[[int], bool]] = [c.instance_free for c in peers if callable(getattr(c, "instance_free", None))]
    locks = Path("~/.cache/nexus/px4-instances").expanduser()
    locks.mkdir(parents=True, exist_ok=True)
    for instance in range(COUNT):
        lock = open(locks / f"{instance}.lock", "w")  # the claim closes it at its stop
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock.close()
            continue
        if all(free(instance) for free in checks):
            return InstanceClaim(instance, lock)
        lock.close()
    raise RuntimeError(f"no free instance on this machine: {COUNT} are in use")
