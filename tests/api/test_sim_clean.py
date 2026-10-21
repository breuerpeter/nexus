"""A run from the command line asked to start clean: its PX4 boots from an empty store.

The docker daemon is the system boundary: the stand-in daemon of the PX4 peer's tests records the containers a run
starts, under a home folder in the test's folder, so a start empties no real store. Each test builds the fixture
vehicle with PX4 declared on the Warp CPU backend and closes it, so PX4's start runs and no PX4 boots. Skipped without
newton or pxr.
"""

import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

from nexus_uav._src.api.args import sim_argparser
from nexus_uav._src.api.sim import Sim
from tests.peers.px4_sitl.test_px4_sitl import daemon  # noqa: F401  the stand-in docker daemon
from tests.usd import sensor_vehicle as sv

# PX4's ports are the machine's, so the tests that bind or dial them run on one worker, one at a time.
pytestmark = [pytest.mark.xdist_group("px4_ports"), pytest.mark.usefixtures("warp_cpu")]


@pytest.fixture
def px4_store(daemon, tmp_path, monkeypatch):  # noqa: F811
    """The fixture vehicle with PX4 declared on airframe `astro_max`, over a stand-in PX4 tree, and the store of
    `astro_max` on instance 0 under the test's home, holding the `parameters.bson` and `dataman` an earlier run left.
    Returns the vehicle's path and the store.
    """
    tree = tmp_path / "px4"
    tree.mkdir()
    (tree / "Makefile").write_text("px4_sitl:\n")
    monkeypatch.setenv("NEXUS_PX4_SITL_REF", str(tree))
    store = tmp_path / "home" / ".cache" / "nexus" / "px4_sitl" / "store" / "astro_max" / "0"
    store.mkdir(parents=True)
    (store / "parameters.bson").write_bytes(b"left by an earlier run")
    (store / "dataman").write_bytes(b"left by an earlier run")
    return sv.vehicle(tmp_path, px4=True), store


def test_a_run_from_the_command_line_can_be_asked_to_start_clean(px4_store, monkeypatch):
    """A run from the command line can be asked to start clean.

    Given a store on `astro_max`, instance 0, outside the test suite's own default, when the command line's
    arguments with `--clean` build that vehicle, then the store holds only the empty `log` the run's folder mounts
    over when PX4 boots.
    """
    monkeypatch.delenv("NEXUS_CLEAN", raising=False)
    vehicle, store = px4_store
    args = sim_argparser().parse_args(["--vehicle", vehicle, "--scene", sv.SCENE, "--device", "cpu", "--clean"])

    with Sim.from_args(args):
        pass

    assert sorted(p.name for p in store.iterdir()) == ["log"]
