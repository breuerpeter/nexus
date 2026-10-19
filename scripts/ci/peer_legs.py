"""The peer legs a change flies: which peers a diff touches, and what each peer's box runs.

A peer is a folder under ``nexus_uav/_src/peers/`` that holds a ``pin.ref``, read from the tree, so a new peer
needs no edit here. A path under ``peers/<name>/`` touches that peer, a path in ``peers/`` itself touches every
peer, and any other path touches none. A touched peer's box runs the tests marked with the peer's name, when
the suite registers such a marker, and flies the example ``EXAMPLES`` maps the peer to, when it names one. A
touched peer with neither has nothing to fly: its box never starts, and the report says so.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

PEERS = Path("nexus_uav/_src/peers")

EXAMPLES: dict[str, str] = {}
"""The example each peer's box flies, by the peer's folder name. A peer with no entry flies no example."""


@dataclass(frozen=True)
class Leg:
    """What one touched peer's box runs: the marker of its tests and the example it flies, each ``None`` for none."""

    peer: str
    tests: str | None
    example: str | None


def peers(root: Path) -> list[str]:
    """The name of each peer in the tree at ``root``: each folder under ``nexus_uav/_src/peers/`` that holds a ``pin.ref``."""


def touched(changed: Iterable[str], root: Path) -> list[str]:
    """The peers of the tree at ``root`` that a diff of the repo-relative paths ``changed`` touches, sorted."""


def legs(changed: Iterable[str], root: Path) -> list[Leg]:
    """One ``Leg`` per peer ``changed`` touches in the tree at ``root``, sorted by peer."""


def report(legs: Iterable[Leg]) -> str:
    """One line per leg: what its box runs, or that the peer has nothing to fly."""
