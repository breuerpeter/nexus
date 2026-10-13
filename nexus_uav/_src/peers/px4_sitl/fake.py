"""The fake PX4 Software In The Loop (SITL) peer: a stand-in that speaks the Hardware In The Loop (HIL) link and
starts no process.

It dials the run's HIL port as PX4 does and answers each ``HIL_SENSOR`` with one fixed
``HIL_ACTUATOR_CONTROLS``, over the same MAVLink lockstep. So the controller and the loop run
exactly as they do against PX4, and a test proves their side of the link, its encoding, its stage
order and its error paths, with no PX4 tree, no image and no container. It proves nothing about PX4
itself, and it answers no other link: a run against it has no offboard link, and its port map lists
none.

A test sends the PX4 SITL peer to this class with a row of its own registry, `NexusPx4SitlAPI`.
"""

from __future__ import annotations

import collections
import os
import select
import socket
import threading

# Set the MAVLink dialect before importing mavutil, as the controller does.
os.environ.setdefault("MAVLINK20", "1")
os.environ.setdefault("MAVLINK_DIALECT", "common")

from pymavlink import mavutil

from nexus_uav._src.core.ports import PortMap

from . import HIL_PORT, checkout
from .instance import hil_free

HOVER = 0.5
"""The command on every channel: a fixed throttle, not tuned to the vehicle's weight."""


class Px4Fake:
    """A stand-in for one run's PX4 SITL container, on the peer contract.

    Args:
        instance: The run's instance: the fake dials the HIL port ``HIL_PORT + instance`` and speaks
            as system ``instance + 1``, as PX4 does.
        **run: The schema's airframe and the run's other values, its pin and its folder, which the fake
            needs none of.
    """

    name = checkout.NAME
    check_pin = staticmethod(checkout.check)
    instance_free = staticmethod(hil_free)

    def __init__(self, *, instance: int, **run) -> None:
        self.instance = instance
        self.received: collections.Counter[str] = collections.Counter()
        """How many of each MAVLink message the fake received, by type."""
        self.last: dict[str, object] = {}
        """The last MAVLink message of each type the fake received, by type."""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def ports(self) -> PortMap:
        """No offboard link: the fake answers the HIL link alone, so a lookup of the offboard link says why."""
        return PortMap(missing={"offboard": "this run flies the fake PX4, which answers only the HIL link"})

    def start(self) -> None:
        """Start dialing the HIL port in the background; the fake answers once the run listens."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._serve, name=f"px4-fake-{self.instance}", daemon=True)
        self._thread.start()

    def alive(self) -> bool:
        """Whether the fake still serves the link."""
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        """Hang up the link and end the fake. Idempotent."""
        self._stop.set()
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
                    self.last[kind] = msg
                    if kind == "HIL_SENSOR":
                        mav.hil_actuator_controls_send(msg.time_usec, [HOVER] * 16, 0, 0)
        except OSError:
            pass
        finally:
            sock.close()
