"""Whether a run's instance is free for PX4 on this machine.

The run claims one instance for every peer it starts, through :mod:`nexus_uav._src.peers.instance`, and asks
each peer's class whether the instance is free for it. PX4 numbers every link from its instance, and the sim
listens on its Hardware In The Loop (HIL) port, so an instance whose HIL port another process listens on is
busy for PX4.
"""

from __future__ import annotations

import socket

from . import HIL_PORT


def hil_free(instance: int) -> bool:
    """Whether no process listens on the HIL port of `instance`, as the HIL server binds it: a connection in
    TIME_WAIT doesn't block it.
    """
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", HIL_PORT + instance))
        except OSError:
            return False
    return True
