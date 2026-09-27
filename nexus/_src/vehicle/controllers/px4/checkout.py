"""How the PX4 controller reaches its PX4 tree.

The controller ships the pin, ``px4.ref`` beside this module: line one is ``repo@sha``, the
PX4-Autopilot tree it flies, and line two the branch ``px4-bump.yml`` tracks. A project overrides it
with a ``nexus.px4.ref`` of the same form beside its catalog, so every vehicle of the project flies
one tree and a bump is one edit. On first use the controller fetches the pinned commit into a folder
it owns, ``~/.cache/nexus/px4/<sha>``, and the PX4 Software In The Loop (SITL) peer builds it there.
``$PX4_DIR`` names a checkout of your own that overrides the fetch, for work on PX4 itself.

The fetch is by commit, depth one, with submodules, and it applies the interim airframe patch the
pinned branch doesn't carry yet: ``EKF2_MAG_TYPE 6`` on ``80000_none_astro_max``, heading from the
magnetometer at initialization only, so the sim's magnetometer doesn't fault continuous fusion,
see #87 and PX4 PR #27706. A tree that carries the line already keeps it as it stands.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from nexus._src.core import logger

PROJECT_PIN = "nexus.px4.ref"  # a project's pin, beside its nexus.registry.yaml
_SHIPPED_PIN = Path(__file__).with_name("px4.ref")
_AIRFRAME = Path("ROMFS/px4fmu_common/init.d-posix/airframes/80000_none_astro_max")
_MAG_PATCH = "param set-default EKF2_MAG_TYPE 6"


@dataclass(frozen=True, slots=True)
class Pin:
    """One PX4 tree: a GitHub repository, ``owner/name``, and a commit."""

    repo: str
    sha: str

    @property
    def url(self) -> str:
        return f"https://github.com/{self.repo}.git"


def pin(catalog: Path | None = None) -> Pin:
    """The pin the controller flies: the project's ``nexus.px4.ref`` beside ``catalog`` when there is
    one, else the pin this package ships.

    Args:
        catalog: The project catalog the run resolved against, or ``None`` for the bundled one.
    """
    source = _SHIPPED_PIN
    if catalog is not None and (candidate := Path(catalog).parent / PROJECT_PIN).is_file():
        source = candidate
    repo, sha = source.read_text().splitlines()[0].strip().split("@", 1)
    return Pin(repo, sha)


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


def tree(catalog: Path | None = None, *, fetch_missing: bool = True) -> Path | None:
    """The PX4 tree the controller flies: ``$PX4_DIR`` when set, else the pinned commit in the folder
    the controller owns, fetched on first use.

    Args:
        catalog: The project catalog the run resolved against, whose pin file wins over the shipped one.
        fetch_missing: With ``False``, a pinned tree not yet on this machine is ``None`` rather than
            fetched: for a check that must use no network.

    Returns:
        The tree, or ``None`` when ``fetch_missing`` is off and no tree is on this machine.
    """
    if env := os.environ.get("PX4_DIR"):
        return Path(env)
    p = pin(catalog)
    dest = Path.home() / ".cache" / "nexus" / "px4" / p.sha
    if not (dest / "Makefile").exists():
        if not fetch_missing:
            return None
        fetch(dest, url=p.url, sha=p.sha)
    return dest
