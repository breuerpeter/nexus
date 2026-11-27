"""The runner of the Micro XRCE-DDS agent peer: one agent in a container, for one run.

The image builds the agent from the commit the pin names, once per machine and per pin. The container runs on
host networking, as PX4's does, so PX4 reaches the agent on the loopback address, and a writer on the host
finds the agent's topics on the Data Distribution Service (DDS).
"""

from __future__ import annotations

import re
import socket
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from nexus_uav._src.peers import pins
from nexus_uav._src.peers.containers import ContainerPeer, ensure_image
from nexus_uav._src.peers.pins import PeerPin

from . import AGENT_PORT

if TYPE_CHECKING:
    from nexus_uav._src.config.catalog import CatalogSource

NAME = "xrce_agent"  # the peer's folder name, which names its pin file, its variable and its container
IMAGE = "nexus-xrce-agent"
IMAGE_DIR = Path(__file__).with_name("image")
TRANSPORTS = ("udp4", "udp6")
_PIN = re.compile(r"[\w.-]+/[\w.-]+@[0-9a-f]{7,40}")


def check(pin: PeerPin) -> PeerPin:
    """``pin``, checked: it names a commit of a GitHub repository, ``owner/name@<commit>``.

    Raises:
        ValueError: The pin has another form; the message names its source and the form it takes.
    """
    if not _PIN.fullmatch(pin.ref):
        raise ValueError(
            f"{pin.source}: the agent's pin {pin.ref!r} isn't repo@sha, such as eProsima/Micro-XRCE-DDS-Agent@<commit>"
        )
    return pin


def resolve(sources: Sequence[CatalogSource] = ()) -> PeerPin:
    """The peer's pin, checked, through :func:`nexus_uav._src.peers.pins.resolve`.

    Args:
        sources: The catalog sources the run loaded, in load order; none for a run handed its catalog.

    Raises:
        ValueError: The pin isn't in the form :func:`check` takes, or two installed packages pin it.
    """
    return check(pins.resolve(NAME, sources))


def _image(pin: PeerPin) -> str:
    """The tag of the image that holds the agent ``pin`` names, built when this machine lacks it."""
    repo, sha = pin.ref.split("@", 1)
    return ensure_image(IMAGE_DIR, IMAGE, {"REPO": repo, "SHA": sha})


def prepare(pin: PeerPin | None = None) -> str:
    """Build the agent's image the way a run's first start does, and return its tag.

    For a caller that wants the cold work done outside a run's budget: CI before a flight or a test, or a
    machine's first setup. ``None`` takes the pin with no project catalog.
    """
    return _image(pin or resolve())


class XrceAgent(ContainerPeer):
    """One Micro XRCE-DDS agent container for one run, a :class:`ContainerPeer`.

    Args:
        transport: The transport the agent listens on, ``udp4`` or ``udp6``: the ``nexus:transport`` of the
            vehicle's ``NexusXrceAgentAPI``.
        pin: The peer's pin, checked by :meth:`check_pin`.
        instance: The run's instance: the agent listens on ``AGENT_PORT + instance``.
        log_dir: The run's folder: the agent's console is ``xrce_agent/console.log`` there.
        **run: The run's other values, which the peer needs none of.

    Raises:
        ValueError: The transport is none the agent's port takes; raised as the build makes the peer.
    """

    name = NAME
    check_pin = staticmethod(check)
    clock = "on_request"  # the agent keeps no time of its own: it forwards a message when one end sends it

    def __init__(self, *, transport: str = "udp4", pin: PeerPin, instance: int, log_dir: Path, **run) -> None:
        if transport not in TRANSPORTS:
            raise ValueError(f"NexusXrceAgentAPI authors nexus:transport {transport!r}; it is one of {TRANSPORTS}")
        super().__init__(instance=instance, log_dir=log_dir)
        self.pin = pin
        self.transport = transport

    def run_options(self) -> dict:
        """The agent on its instance's port, on host networking."""
        return {
            "image": _image(self.pin),
            "command": ["MicroXRCEAgent", self.transport, "-p", str(AGENT_PORT + self.instance)],
            "network_mode": "host",
        }

    @staticmethod
    def instance_free(instance: int) -> bool:
        """Whether ``instance`` is free for the agent: no process holds its port."""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            try:
                s.bind(("0.0.0.0", AGENT_PORT + instance))
            except OSError:
                return False
        return True
