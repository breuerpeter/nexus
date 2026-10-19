"""The peer legs a change flies: which peers a diff touches, and what each touched peer's box runs.

A peer is a package folder under ``nexus_uav/_src/peers/``, read from the tree, so a new peer needs no edit
here. A path under ``peers/<name>/`` touches that peer, a path in ``peers/`` itself touches every peer, and any
other path touches none. A touched peer's box runs the tests marked with the peer's name, when ``pyproject.toml``
registers that marker, and flies the example ``EXAMPLES`` maps the peer to, when it names one. A touched peer
with neither has nothing to fly: its box never starts, and the report says so. A push to main runs every
peer's tests, one box each, and no example: main's gpu-examples flies the examples.

    python3 scripts/ci/peer_legs.py --changed <file>   # a pull request: its changed paths, one per line
    python3 scripts/ci/peer_legs.py --all              # a push to main

Either prints one JSON for the workflow: ``matrix``, the ``include`` rows of the legs with something to fly,
each a ``peer``, its ``tests`` marker and its ``example``, and ``report``, one line per peer.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PEERS = Path("nexus_uav/_src/peers")

EXAMPLES: dict[str, str] = {"px4_sitl": "px4_sitl"}
"""The example each peer's box flies, by the peer's folder name: a CI choice of what to spend a box on. A peer
with no entry flies no example."""


@dataclass(frozen=True)
class Leg:
    """What one peer's box runs: the marker of its tests and the example it flies, each ``None`` for none."""

    peer: str
    tests: str | None
    example: str | None

    @property
    def flies(self) -> bool:
        """Whether the box has anything to run."""
        return self.tests is not None or self.example is not None


def peers(root: Path) -> list[str]:
    """The name of each peer in the tree at ``root``: each package folder under ``nexus_uav/_src/peers/``."""
    return sorted(folder.name for folder in (root / PEERS).iterdir() if (folder / "__init__.py").is_file())


def touched(changed: Iterable[str], root: Path) -> list[str]:
    """The peers of the tree at ``root`` that a diff of the repo-relative paths ``changed`` touches, sorted."""
    names = peers(root)
    hit: set[str] = set()
    for path in changed:
        parts = Path(path).parts
        if parts[: len(PEERS.parts)] != PEERS.parts:
            continue
        below = parts[len(PEERS.parts) :]
        if len(below) == 1:
            return names  # a module of the peers folder itself: the contract, the pins or the instance claim
        if below and below[0] in names:
            hit.add(below[0])
    return sorted(hit)


def markers(root: Path) -> set[str]:
    """The markers the suite at ``root`` registers with pytest, by name."""
    config = tomllib.loads((root / "pyproject.toml").read_text())
    entries = config.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("markers", [])
    return {entry.split(":", 1)[0].strip() for entry in entries}


def legs(changed: Iterable[str], root: Path) -> list[Leg]:
    """One ``Leg`` per peer ``changed`` touches in the tree at ``root``, sorted by peer."""
    marked = markers(root)
    return [
        Leg(peer=name, tests=name if name in marked else None, example=EXAMPLES.get(name))
        for name in touched(changed, root)
    ]


def every(root: Path) -> list[Leg]:
    """One ``Leg`` per peer of the tree at ``root`` with tests, the tests alone: for a push to main, where
    gpu-examples flies the examples one box each.
    """
    marked = markers(root)
    return [Leg(peer=name, tests=name, example=None) for name in peers(root) if name in marked]


def report(legs: Iterable[Leg]) -> str:
    """One line per leg: what its box runs, or that the peer has nothing to fly."""
    lines = []
    for leg in legs:
        if not leg.flies:
            lines.append(f"{leg.peer}: touched, no test marked {leg.peer} and no example, nothing to fly")
            continue
        tests = f"tests marked {leg.tests}" if leg.tests else "no tests"
        example = f"example {leg.example}" if leg.example else "no example"
        lines.append(f"{leg.peer}: {tests}, {example}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--changed", metavar="FILE", help="a pull request's changed paths, one per line")
    what.add_argument("--all", action="store_true", help="every peer with tests, the tests alone: a push to main")
    args = ap.parse_args()

    if args.all:
        plan = every(ROOT)
    else:
        changed = [line.strip() for line in Path(args.changed).read_text().splitlines() if line.strip()]
        plan = legs(changed, ROOT)
    rows = [asdict(leg) for leg in plan if leg.flies]
    print(json.dumps({"matrix": {"include": rows}, "report": report(plan)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
