"""The fake PX4 Software In The Loop (SITL) peer: a stand-in that speaks the Hardware In The Loop (HIL) link and
starts no process.

It dials the run's HIL port as PX4 does and answers each ``HIL_SENSOR`` with one fixed
``HIL_ACTUATOR_CONTROLS``, over the same MAVLink lockstep. So the controller and the loop run
exactly as they do against PX4, and a test proves their side of the link, its encoding, its stage
order and its error paths, with no PX4 tree, no image and no container. It proves nothing about PX4
itself, and it answers no other link: a run against it has no operator link.

A test sends the PX4 SITL peer to this class through the builder's peer mapping.
"""

from __future__ import annotations

import collections
import os
import select
import socket
import threading
from typing import IO

# Set the MAVLink dialect before importing mavutil, as the controller does.
os.environ.setdefault("MAVLINK20", "1")
os.environ.setdefault("MAVLINK_DIALECT", "common")

from pymavlink import mavutil

from . import HIL_PORT
from .instance import claim as _claim

HOVER = 0.5
"""The command on every channel: a fixed throttle, not tuned to the vehicle's weight."""


class Px4Fake:
    """A stand-in for one run's PX4 SITL container, on the peer contract.

    Args:
        instance: PX4's SITL instance: the fake dials the HIL port ``HIL_PORT + instance`` and speaks
            as system ``instance + 1``, as PX4 does.
        claim: The open lock file that holds ``instance`` for this run, which the stop releases, as
            the real peer's does; ``None`` for a fake whose instance no lock holds.
        **run: The other run facts the build hands a PX4 peer, the catalog, the airframe, the
            container's name and its console log, which the fake needs none of.
    """

    def __init__(self, *, instance: int, claim: IO | None = None, **run) -> None:
        self.instance = instance
        self._claim = claim
        self.received: collections.Counter[str] = collections.Counter()
        """How many of each MAVLink message the fake received, by type."""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def claim_instance() -> tuple[int, IO]:
        """The lowest PX4 instance free on this machine and the lock file that holds it, as the real
        peer claims one: two runs against fakes take two HIL ports too.
        """
        return _claim()

    def start(self) -> None:
        """Start dialing the HIL port in the background; the fake answers once the run listens."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._serve, name=f"px4-fake-{self.instance}", daemon=True)
        self._thread.start()

    def alive(self) -> bool:
        """Whether the fake still serves the link."""
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        """Hang up the link, end the fake and release its instance. Idempotent."""
        self._stop.set()
        if self._claim is not None:
            self._claim.close()
            self._claim = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _dial(self) -> socket.socket | None:
        """A socket connected to the run's HIL port, dialing again until the run listens or the fake stops."""
        while not self._stop.is_set():
            try:
                return socket.create_connection(("127.0.0.1", HIL_PORT + self.instance), timeout=1.0)
            except OSError:
                self._stop.wait(0.05)
        return None

    def _serve(self) -> None:
        sock = self._dial()
        if sock is None:
            return
        mav = mavutil.mavlink.MAVLink(sock.makefile("wb", buffering=0), srcSystem=self.instance + 1, srcComponent=1)
        try:
            while not self._stop.is_set():
                if not select.select([sock], [], [], 0.1)[0]:
                    continue
                data = sock.recv(65536)
                if not data:
                    return  # the run closed the link
                for msg in mav.parse_buffer(data) or []:
                    kind = msg.get_type()
                    self.received[kind] += 1
                    if kind == "HIL_SENSOR":
                        mav.hil_actuator_controls_send(msg.time_usec, [HOVER] * 16, 0, 0)
        except OSError:
            pass
        finally:
            sock.close()
