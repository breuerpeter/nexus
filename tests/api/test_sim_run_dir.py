"""``Sim(..., run_dir=)`` names the folder a run writes its files to, #320.

A real build of the fixture vehicle in ``tests/usd/sensor_vehicle.py`` on the Warp CPU backend, through
``Sim``, with the home folder in the test's folder. Skipped without newton or pxr.
"""

from pathlib import Path

import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import nexus_uav as nx
from tests.usd import sensor_vehicle as sv

pytestmark = pytest.mark.usefixtures("warp_cpu")


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """A home folder in the test's folder, so the default run folder lands there."""
    folder = tmp_path / "home"
    monkeypatch.setenv("HOME", str(folder))
    return folder


def _fly(vehicle: str, registry: nx.Registry | None = None, **kw) -> dict:
    """Run `vehicle` in the fixture scene on the CPU device with recording on, through its start, and return
    its artifacts.
    """
    with nx.Sim(vehicle, scene=sv.SCENE, device="cpu", log=True, registry=registry or sv.components(), **kw) as sim:
        sim.start(timeout=5.0)
    return sim.artifacts()


def _writing_peer() -> type:
    """A stand-in class of a project's own peer that writes `radio.log` into the run's folder the build gives it."""

    class Writing(sv.radio()):
        def start(self) -> None:
            super().start()
            (Path(self.run["log_dir"]) / "radio.log").write_text("on air\n")

    return Writing


def test_a_run_whose_run_dir_does_not_exist_yet_creates_it_and_writes_its_recording_there(home, tmp_path):
    """A run whose `run_dir` names a folder that doesn't exist yet creates it and writes its recording there,
    and `sim.artifacts()` names it there.

    Given the home folder in a temp folder, the fixture vehicle on stand-ins and no folder at `d`, when
    `Sim(..., log=True, run_dir=d)` runs and closes, then `d/nexus.rrd` exists, `sim.artifacts()["rrd"]` is that
    path, and `~/.cache/nexus/logs` holds no folder.
    """
    d = tmp_path / "results" / "run"
    artifacts = _fly(sv.vehicle(tmp_path), run_dir=d)

    logs = home / ".cache" / "nexus" / "logs"
    seen = ((d / "nexus.rrd").is_file(), artifacts["rrd"], sorted(logs.glob("*")))
    assert seen == (True, str(tmp_path / "results" / "run" / "nexus.rrd"), [])


def test_an_empty_folder_run_dir_names_takes_the_runs_files(home, tmp_path):
    """An empty folder that `run_dir` names takes the run's files.

    Given an empty `d`, when the run closes, then `d/nexus.rrd` exists.
    """
    d = tmp_path / "run"
    d.mkdir()
    _fly(sv.vehicle(tmp_path), run_dir=d)

    assert (d / "nexus.rrd").is_file()


def test_a_relative_run_dir_names_a_folder_under_the_working_folder(home, tmp_path, monkeypatch):
    """A relative `run_dir` names a folder under the working folder.

    Given the working folder at a temp folder `w` and the fixture vehicle on stand-ins, when
    `Sim(..., log=True, run_dir="run")` runs and closes, then `w/run/nexus.rrd` exists and
    `sim.artifacts()["rrd"]` is its absolute path.
    """
    w = tmp_path / "work"
    w.mkdir()
    monkeypatch.chdir(w)
    artifacts = _fly(sv.vehicle(tmp_path), run_dir="run")

    assert ((w / "run" / "nexus.rrd").is_file(), artifacts["rrd"]) == (
        True,
        str(tmp_path / "work" / "run" / "nexus.rrd"),
    )


def test_a_run_dir_that_holds_a_file_fails_the_run_before_any_peer_starts_and_keeps_the_file(home, tmp_path):
    """A folder that `run_dir` names and that holds a file fails the run before any peer starts, naming the folder,
    and the file stays.

    Given `d` holding `old.txt` and a vehicle that declares a stand-in peer, when `Sim(..., run_dir=d)` enters,
    then it raises an error that names `d`, the peer never started, and `d` holds `old.txt` alone.
    """
    d = tmp_path / "run"
    d.mkdir()
    (d / "old.txt").write_text("an earlier run's\n")
    peer = sv.radio()
    registry = sv.components(StandInPeerAPI=peer)
    try:
        _fly(sv.vehicle(tmp_path, root=sv.companion()), registry, run_dir=d)
        named = False
    except Exception as exc:
        named = str(d) in str(exc)

    assert (named, peer.events, sorted(p.name for p in d.iterdir())) == (True, [], ["old.txt"])


def test_each_peer_of_a_run_with_run_dir_writes_its_files_in_that_folder(home, tmp_path):
    """Each peer of a run with `run_dir` writes its files in that folder.

    Given a stand-in peer that writes a file into the run's folder the build gives it, when
    `Sim(..., run_dir=d, registry=...)` runs and closes, then the file is in `d`.
    """
    d = tmp_path / "run"
    registry = sv.components(StandInPeerAPI=_writing_peer())
    _fly(sv.vehicle(tmp_path, root=sv.companion()), registry, run_dir=d)

    assert (d / "radio.log").is_file()


def test_a_run_with_no_run_dir_writes_its_files_to_a_new_folder_under_the_logs_cache_one_per_run(home, tmp_path):
    """A run with no `run_dir` writes its files to a new folder under `~/.cache/nexus/logs`, one per run.

    Given the home folder in a temp folder, when two runs on stand-ins close with `log=True`, then each `nexus.rrd` sits in
    its own folder under `~/.cache/nexus/logs`.
    """
    vehicle = sv.vehicle(tmp_path)
    rrds = [Path(_fly(vehicle)["rrd"]) for _ in range(2)]

    logs = home / ".cache" / "nexus" / "logs"
    seen = ([(r.name, r.parent.parent, r.is_file()) for r in rrds], len({r.parent for r in rrds}))
    assert seen == ([("nexus.rrd", logs, True)] * 2, 2)
