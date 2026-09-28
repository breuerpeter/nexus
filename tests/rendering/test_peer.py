"""The Kit container's spec: the image it runs, what it reads, and who owns what it writes."""

import io
import os
import urllib.request

import pytest

from nexus._src.rendering.peer import KitPeer, KitPeerError, run_script

ISAAC_SIM = "nvcr.io/nvidia/isaac-sim:6.0.1"
CESIUM_SCENE = '#usda 1.0\n\ndef CesiumTilesetPrim "Cesium_World_Terrain"\n{\n}\n'
EMPTY_SCENE = '#usda 1.0\n\ndef Xform "World"\n{\n}\n'


def _usd(folder, name="v.usda", text="#usda 1.0\n"):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(text)
    return path


def _network(monkeypatch, body=b"", requested=None):
    """The network, answering every URL with ``body`` and noting each URL in ``requested``."""

    def urlopen(url, *args, **kwargs):
        if requested is not None:
            requested.append(getattr(url, "full_url", url))
        return io.BytesIO(body)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)


def test_the_container_reads_the_asset_cache_at_its_host_path(daemon, tmp_path):
    cache = tmp_path / "cache"
    KitPeer([_usd(cache / "0a1b")], cache_dir=cache).start()
    assert daemon.runs[0]["volumes"][str(cache)] == {"bind": str(cache), "mode": "ro"}


def test_a_local_usd_is_read_from_its_own_folder(daemon, tmp_path):
    project = tmp_path / "project"
    KitPeer([_usd(project, "cam.usda")], cache_dir=tmp_path / "cache").start()
    assert daemon.runs[0]["volumes"][str(project)] == {"bind": str(project), "mode": "ro"}


def test_every_folder_the_container_writes_exists_as_the_user_before_it_starts(daemon, tmp_path):
    """Docker creates a missing bind source owned by root, which a later host run can't write."""
    KitPeer([_usd(tmp_path / "project")], cache_dir=tmp_path / "cache").start()
    run = daemon.runs[0]
    writable = [src for src, spec in run["volumes"].items() if spec["mode"] == "rw"]
    assert writable and {run["seen"][src] for src in writable} == {(True, os.getuid())}


def test_the_container_runs_as_the_user(daemon, tmp_path):
    """So every file it writes into a mounted cache belongs to the user."""
    KitPeer([_usd(tmp_path / "project")], cache_dir=tmp_path / "cache").start()
    assert daemon.runs[0]["user"] == f"{os.getuid()}:{os.getgid()}"


def test_the_container_environment_holds_no_ion_token(daemon, tmp_path, monkeypatch):
    """Kit prints its whole environment into the console at every boot, and the host keeps the console as a log."""
    monkeypatch.setenv("CESIUM_ION_TOKEN", "ion-secret")
    KitPeer([_usd(tmp_path / "project")], cache_dir=tmp_path / "cache").start()
    assert "ion-secret" not in daemon.runs[0]["environment"].values()


def test_a_kit_script_container_environment_holds_no_ion_token(daemon, tmp_path, monkeypatch):
    monkeypatch.setenv("CESIUM_ION_TOKEN", "ion-secret")
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "convert.py"
    script.write_text("")
    run_script([str(script)])
    assert "ion-secret" not in daemon.runs[0]["environment"].values()


def test_the_kit_container_runs_nvidias_image_as_pulled(daemon, tmp_path):
    """The Kit container runs NVIDIA's image as pulled."""
    KitPeer([_usd(tmp_path / "project")], cache_dir=tmp_path / "cache").start()
    assert (daemon.runs[0]["image"], daemon.builds) == (ISAAC_SIM, [])


def test_a_machine_that_lacks_the_image_pulls_it_once(daemon, tmp_path):
    """A machine that lacks the image pulls it once; a machine that holds it pulls nothing."""
    daemon.held = set()
    vehicle = _usd(tmp_path / "project")
    for _ in range(2):
        KitPeer([vehicle], cache_dir=tmp_path / "cache").start()
    assert (daemon.pulls, daemon.builds) == ([ISAAC_SIM], [])


def test_a_failed_pull_ends_the_run_with_the_image_and_the_cause(daemon, tmp_path):
    """A failed pull ends the run before the flight, with a message that names the image and the cause."""
    daemon.held = set()
    daemon.pull_error = "toomanyrequests: retry later"
    with pytest.raises(KitPeerError) as err:
        KitPeer([_usd(tmp_path / "project")], cache_dir=tmp_path / "cache").start()
    assert (ISAAC_SIM in str(err.value), "toomanyrequests" in str(err.value), daemon.runs) == (True, True, [])


def test_a_cesium_download_that_misses_its_sha256_fails_before_kit_starts(daemon, tmp_path, monkeypatch):
    """A Cesium download whose bytes miss the pinned sha256 fails the run before Kit starts."""
    _network(monkeypatch, body=b"not the Cesium for Omniverse release")
    project = tmp_path / "project"
    files = [_usd(project, "cam.usda"), _usd(project, "cesium.usda", CESIUM_SCENE)]
    with pytest.raises(KitPeerError, match="mismatch"):
        KitPeer(files, cache_dir=tmp_path / "cache").start()
    assert daemon.runs == []


def test_a_scene_with_no_cesium_tileset_fetches_no_cesium(daemon, tmp_path, monkeypatch):
    """A run on a scene with no Cesium tileset fetches no Cesium."""
    requested: list[str] = []
    _network(monkeypatch, requested=requested)
    project = tmp_path / "project"
    files = [_usd(project, "cam.usda"), _usd(project, "empty.usda", EMPTY_SCENE)]
    KitPeer(files, cache_dir=tmp_path / "cache").start()
    assert [url for url in requested if "cesium" in url.lower()] == []
