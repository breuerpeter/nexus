"""How the PX4 peer reaches its PX4 tree: the fetch of the pinned commit into a folder it owns, with
the interim airframe patch. ``tests/peers/test_pins.py`` covers where the pin comes from.

A git repository in the test's folder stands in for PX4-Autopilot, so a fetch touches no network.
"""

import subprocess
from pathlib import Path

from nexus_uav._src.peers.px4_sitl import checkout

AIRFRAME = Path("ROMFS/px4fmu_common/init.d-posix/airframes/80000_none_astro_max")
MAG_PATCH = "param set-default EKF2_MAG_TYPE 6"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _repository(tmp_path) -> tuple[str, str]:
    """A stand-in PX4-Autopilot: one commit carrying the airframe without the patch. Returns its URL
    and that commit's sha.
    """
    repo = tmp_path / "px4-autopilot"
    (repo / AIRFRAME).parent.mkdir(parents=True)
    (repo / AIRFRAME).write_text("#!/bin/sh\n. ${R}etc/init.d/rc.mc_defaults\n")
    (repo / "Makefile").write_text("px4_sitl:\n\ttrue\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "the pinned tree")
    _git(repo, "config", "uploadpack.allowAnySHA1InWant", "true")  # a fetch by sha, as from GitHub
    return str(repo), _git(repo, "rev-parse", "HEAD")


def test_the_fetched_tree_is_the_pinned_commit_and_carries_the_interim_airframe_patch_once(tmp_path):
    """The fetched tree is the pinned commit and carries the interim airframe patch once.

    Given a fetch into an empty folder against a stand-in repository, when reading the airframe file,
    then the checkout is at the pin and ``param set-default EKF2_MAG_TYPE 6`` occurs once, and once
    again after a second fetch.
    """
    url, sha = _repository(tmp_path)
    dest = tmp_path / "checkout"

    checkout.fetch(dest, url=url, sha=sha)
    first = (dest / AIRFRAME).read_text().count(MAG_PATCH)
    checkout.fetch(dest, url=url, sha=sha)
    second = (dest / AIRFRAME).read_text().count(MAG_PATCH)

    assert (_git(dest, "rev-parse", "HEAD"), first, second) == (sha, 1, 1)
