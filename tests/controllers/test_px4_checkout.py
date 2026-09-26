"""How the PX4 controller reaches its PX4 tree: the fetch of the pinned commit into a folder it owns,
with the interim airframe patch; ``$PX4_DIR`` as the checkout that overrides the fetch; and the pin a
project keeps beside its catalog over the one the controller ships.

A git repository in the test's folder stands in for PX4-Autopilot, so a fetch touches no network.
"""

import importlib.resources
import subprocess
from pathlib import Path

from nexus._src.vehicle.controllers.px4 import checkout

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

    Given a fetch into an empty folder against a stand-in repository, when the airframe file is read,
    then the checkout is at the pin and ``param set-default EKF2_MAG_TYPE 6`` appears once, and once
    again after a second fetch.
    """
    url, sha = _repository(tmp_path)
    dest = tmp_path / "checkout"

    checkout.fetch(dest, url=url, sha=sha)
    first = (dest / AIRFRAME).read_text().count(MAG_PATCH)
    checkout.fetch(dest, url=url, sha=sha)
    second = (dest / AIRFRAME).read_text().count(MAG_PATCH)

    assert (_git(dest, "rev-parse", "HEAD"), first, second) == (sha, 1, 1)


def test_px4_dir_names_a_checkout_that_overrides_the_fetch(tmp_path, monkeypatch):
    """``PX4_DIR`` names a checkout that overrides the fetch.

    Given ``PX4_DIR`` set to a folder, when the controller resolves its PX4 tree, then it is that
    folder and no fetch runs.
    """
    folder = tmp_path / "px4"
    folder.mkdir()
    home = tmp_path / "home"
    monkeypatch.setenv("PX4_DIR", str(folder))
    monkeypatch.setenv("HOME", str(home))

    tree = checkout.tree()

    assert (Path(tree), home.exists()) == (folder, False)


def test_a_projects_pin_file_beside_its_catalog_overrides_the_controllers_pin(tmp_path):
    """A project's pin file beside its catalog overrides the controller's pin.

    Given a project catalog with a pin file beside it, when the controller resolves its PX4 tree, then
    it is the project's commit, and with no file it is the controller's.
    """
    catalog = tmp_path / "nexus.registry.yaml"
    catalog.write_text("vehicles: []\n")
    (tmp_path / "nexus.px4.ref").write_text("example/px4@" + "a" * 40 + "\n")
    shipped = importlib.resources.files("nexus._src.vehicle.controllers.px4").joinpath("px4.ref").read_text()
    shipped_sha = shipped.splitlines()[0].split("@", 1)[1]

    assert (checkout.pin(catalog).sha, checkout.pin(None).sha) == ("a" * 40, shipped_sha)
