"""``sim.artifacts()`` reads a peer's ``artifacts`` as optional, as the ``Peer`` contract states.

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


def test_a_run_with_a_peer_that_names_no_artifacts_still_names_its_recording(tmp_path, monkeypatch):
    """A run with a peer that names no artifacts still names its recording in `sim.artifacts()`.

    Given the fixture vehicle that declares a stand-in peer with no `artifacts`, when `Sim(..., log=True)` runs
    and closes, then `sim.artifacts()` holds the one key `rrd`, the run's `nexus.rrd`.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    registry = sv.components(StandInPeerAPI=sv.radio())
    vehicle = sv.vehicle(tmp_path, root=sv.companion())
    with nx.Sim(vehicle, scene=sv.SCENE, device="cpu", log=True, registry=registry) as sim:
        sim.start(timeout=5.0)
    artifacts = sim.artifacts()

    assert (sorted(artifacts), Path(artifacts["rrd"]).name) == (["rrd"], "nexus.rrd")
