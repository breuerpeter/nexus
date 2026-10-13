"""The ULog a PX4 Software In The Loop (SITL) run names in its artifacts: the file PX4 wrote for that
run, and not one an earlier run left in the PX4 tree.

The module fixture flies two runs in a row through ``nx.Sim``, which builds and launches PX4 from the
tree on this machine, so the tests need docker and a PX4 tree carrying the airframe the sim flies;
they skip without one.
"""

import time
from pathlib import Path

import pytest

pytest.importorskip("pymavlink")
pytest.importorskip("newton")

import nexus_uav as nx
from nexus_uav._src.peers.px4_sitl import checkout

# The tree the run would fly, with no fetch: $PX4_DIR, or the pinned tree once a run fetched it.
_TREE = checkout.tree(fetch_missing=False)
# A mark, not a skip at import, so a run that deselects `px4_sitl` lists none of these tests.
# PX4's ports are the machine's, so the tests that bind or dial them run on one worker, one at a time.
pytestmark = [
    pytest.mark.px4_sitl,
    pytest.mark.xdist_group("px4_ports"),
    pytest.mark.skipif(
        _TREE is None
        or not list((_TREE / "ROMFS" / "px4fmu_common" / "init.d-posix" / "airframes").glob("*_none_astro_max")),
        reason="needs docker and a PX4 tree on this machine, $PX4_DIR or the fetched pin, carrying the none_astro_max airframe",
    ),
]


@pytest.fixture(scope="module")
def two_runs():
    """Two PX4 runs in a row, each its wall-clock start, its wall-clock end and its artifacts."""
    runs = []
    for _ in range(2):
        began = time.time()
        with nx.Sim("astro_max_base", scene="empty") as sim:
            sim.start(timeout=120.0)
            sim.sleep(1.0)
        runs.append((began, time.time(), sim.artifacts()))
    return runs


def _written_during(path, began: float, ended: float) -> bool:
    """Whether ``path`` is an existing ``.ulg`` file last written between ``began`` and ``ended``."""
    return path is not None and path.endswith(".ulg") and _exists(path) and began <= Path(path).stat().st_mtime <= ended


def _exists(path) -> bool:
    """Whether ``path`` names an existing file."""
    return path is not None and Path(path).is_file()


def test_a_px4_run_the_sim_starts_names_the_ulog_px4_wrote_for_it(two_runs):
    """A PX4 run the sim starts names the ULog PX4 wrote for it.

    Given a real PX4 run that flies, when it closes, then `sim.artifacts()["ulog"]` is an existing
    `.ulg` file last written during the run.
    """
    began, ended, artifacts = two_runs[0]
    assert _written_during(artifacts.get("ulog"), began, ended), artifacts


def test_a_run_names_its_own_ulog_not_an_earlier_runs(two_runs):
    """A run names its own ULog, not an earlier run's.

    Given two real PX4 runs in a row, when each closes, then each run's `ulog` names an existing file,
    and the two files differ.
    """
    ulogs = [artifacts.get("ulog") for _, _, artifacts in two_runs]
    assert all(_exists(u) for u in ulogs) and ulogs[0] != ulogs[1], ulogs
