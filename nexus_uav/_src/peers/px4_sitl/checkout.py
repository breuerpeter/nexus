"""How the PX4 peer reaches its PX4 tree.

The peer's pin comes from :mod:`nexus_uav._src.peers.pins`: ``pin.ref`` beside this module, a
package's or a project's ``nexus.px4_sitl.ref`` beside its catalog, or ``$NEXUS_PX4_SITL_REF``. In the shipped file,
line one is ``repo@sha``, the PX4-Autopilot tree it flies, and line two the branch ``px4-bump.yml``
tracks. A pin is ``owner/repo@<commit>``, or the absolute path of a checkout of your own, for work on
PX4 itself. On first use the peer fetches a pinned commit into a folder it owns,
``~/.cache/nexus/px4_sitl/<sha>``, and the PX4 Software In The Loop (SITL) peer builds it there. A
peer builds a checkout where it lies, and the receipt records its checked-out commit.

The fetch is by commit, depth one, with submodules, and it applies the interim airframe patch the
pinned branch doesn't carry yet: ``EKF2_MAG_TYPE 6`` on ``80000_none_astro_max``, heading from the
magnetometer at initialization only, so the sim's magnetometer doesn't fault continuous fusion,
see #87 and PX4 PR #27706. A tree that carries the line already keeps it as it stands.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from nexus_uav._src.core import logger
from nexus_uav._src.peers import pins
from nexus_uav._src.peers.pins import PeerPin

if TYPE_CHECKING:
    from nexus_uav._src.config.catalog import CatalogLayer

NAME = "px4_sitl"  # the peer's folder name, which names its pin file and its variable
_AIRFRAME = Path("ROMFS/px4fmu_common/init.d-posix/airframes/80000_none_astro_max")
_MAG_PATCH = "param set-default EKF2_MAG_TYPE 6"
_PIN = re.compile(r"[\w.-]+/[\w.-]+@[0-9a-f]{7,40}")


@dataclass(frozen=True, slots=True)
class Pin:
    """One PX4 tree: a GitHub repository, ``owner/name``, and a commit."""

    repo: str
    sha: str

    @property
    def url(self) -> str:
        return f"https://github.com/{self.repo}.git"


def parse(pin: PeerPin) -> Pin | Path:
    """The PX4 tree ``pin`` names: a :class:`Pin` for ``owner/repo@<commit>``, or a checkout's path.

    A checkout is a folder; one outside git flies too, and the receipt records no commit for it.

    Raises:
        ValueError: The pin is neither, or names a path with no folder; the message names its source
            and the forms it takes.
    """
    ref = pin.ref
    if ref.startswith(("/", "~")):
        if not (tree := Path(ref).expanduser()).is_dir():
            raise ValueError(
                f"{pin.source}: the PX4 pin {ref!r} names no folder; point it at a checkout, or at repo@sha"
            )
        return tree
    if not _PIN.fullmatch(ref):
        raise ValueError(
            f"{pin.source}: the PX4 pin {ref!r} is neither repo@sha, such as PX4/PX4-Autopilot@<commit>, "
            "nor the absolute path of a checkout"
        )
    repo, sha = ref.split("@", 1)
    return Pin(repo, sha)


def resolve(layers: Sequence[CatalogLayer] = ()) -> PeerPin:
    """The peer's pin, checked, through :func:`nexus_uav._src.peers.pins.resolve`; a checkout's pin
    carries its checked-out commit.

    Args:
        layers: The catalog layers the run loaded, in load order; none for a run handed its catalog.

    Raises:
        ValueError: The pin isn't in the form :func:`parse` takes, or two installed packages pin it.
    """
    return check(pins.resolve(NAME, layers))


def check(pin: PeerPin) -> PeerPin:
    """``pin``, checked against the form :func:`parse` takes; a checkout's pin carries its checked-out commit.

    Raises:
        ValueError: The pin isn't in the form :func:`parse` takes; the message names its source and the forms.
    """
    tree = parse(pin)
    if isinstance(tree, Path):
        head = subprocess.run(
            ["git", "-C", str(tree), "rev-parse", "HEAD"], check=False, capture_output=True, text=True
        )
        pin = pin.model_copy(update={"commit": head.stdout.strip() or None})  # no commit outside git
    return pin


def _git(dest: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(dest), *args], check=True)


def fetch(dest: Path, *, url: str, sha: str) -> None:
    """Fetch commit ``sha`` of the repository the URL names into ``dest``, and apply the interim
    airframe patch. Idempotent: a ``dest`` that already holds a tree keeps it, but for the patch,
    which lands once.

    Raises:
        subprocess.CalledProcessError: A git command failed.
    """
    dest = Path(dest)
    if not (dest / "Makefile").exists():
        logger.info(f"fetching PX4 {sha[:12]} from {url} into {dest}")
        dest.mkdir(parents=True, exist_ok=True)
        _git(dest, "init", "-q")
        _git(dest, "fetch", "-q", "--depth", "1", url, sha)
        _git(dest, "checkout", "-q", "FETCH_HEAD")
        _git(dest, "submodule", "update", "--quiet", "--init", "--recursive")
    airframe = dest / _AIRFRAME
    if airframe.is_file() and "EKF2_MAG_TYPE" not in airframe.read_text():
        with airframe.open("a") as f:
            f.write(f"\n{_MAG_PATCH}\n")
        logger.info(f"patched {airframe.name} with {_MAG_PATCH} (interim, PX4 PR #27706)")


def tree(pin: PeerPin | None = None, *, fetch_missing: bool = True) -> Path | None:
    """The PX4 tree the peer flies: the checkout ``pin`` names, or its pinned commit in the folder the
    peer owns, fetched on first use.

    Args:
        pin: The peer's pin, from :func:`resolve`; ``None`` resolves it with no project catalog.
        fetch_missing: With ``False``, a pinned tree not yet on this machine is ``None`` rather than
            fetched: for a check that must use no network.

    Returns:
        The tree, or ``None`` when ``fetch_missing`` is off and no tree is on this machine.
    """
    p = parse(pin or resolve())
    if isinstance(p, Path):
        return p
    dest = Path.home() / ".cache" / "nexus" / "px4_sitl" / p.sha
    if not (dest / "Makefile").exists():
        if not fetch_missing:
            return None
        fetch(dest, url=p.url, sha=p.sha)
    return dest
