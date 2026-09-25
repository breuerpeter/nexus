"""The Kit container's spec: what it reads, and who owns what it writes."""

import os

from nexus._src.rendering.peer import KitPeer, run_script


def _usd(folder, name="v.usda"):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text("#usda 1.0\n")
    return path


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
