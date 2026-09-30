"""The PX4 instance a run holds on this machine.

A run owns every address its PX4 peer uses, so it picks the instance itself: the lowest one no other
run holds and whose Hardware In The Loop (HIL) port no process listens on. A run holds its instance
with a lock on a file of its own under ``~/.cache/nexus/px4-instances/``, taken without waiting,
which the peer releases at its stop. So two runs that start at once take two instances.
"""

from __future__ import annotations

import fcntl
import socket
from collections.abc import Collection
from pathlib import Path
from typing import IO

from . import HIL_PORT


def claim(skip: Collection[int] = ()) -> tuple[int, IO]:
    """The lowest PX4 instance free on this machine, and the open lock file that holds it.

    Args:
        skip: Instances to pass over whatever their lock says, such as those a live container holds.

    Returns:
        The instance and its lock file; closing the file releases the instance.

    Raises:
        RuntimeError: Every one of 256 instances is in use.
    """
    locks = Path("~/.cache/nexus/px4-instances").expanduser()
    locks.mkdir(parents=True, exist_ok=True)
    for instance in range(256):
        if instance in skip:
            continue
        lock = open(locks / f"{instance}.lock", "w")  # the peer closes it at its stop
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with socket.socket() as s:
                # As the HIL server binds: only a listener blocks it, not a connection in TIME_WAIT.
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("0.0.0.0", HIL_PORT + instance))
        except OSError:
            lock.close()
            continue
        return instance, lock
    raise RuntimeError("no free PX4 instance on this machine: 256 are in use")
