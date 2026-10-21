"""A run the test suite makes starts clean without asking, so no test boots a peer on what an earlier run stored.

The docker daemon is the system boundary, as in ``tests/api/test_sim_clean.py``, whose fixture this module shares.
Skipped without newton or pxr.
"""

import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

from nexus_uav._src.api.sim import Sim
from tests.api.test_sim_clean import px4_store  # noqa: F401  the fixture vehicle and a store an earlier run left
from tests.peers.px4_sitl.test_px4_sitl import daemon  # noqa: F401  the stand-in docker daemon
from tests.usd import sensor_vehicle as sv

# PX4's ports are the machine's, so the tests that bind or dial them run on one worker, one at a time.
pytestmark = [pytest.mark.xdist_group("px4_ports"), pytest.mark.usefixtures("warp_cpu")]


def test_a_run_made_in_the_test_suite_starts_clean_without_asking(px4_store):  # noqa: F811
    """A run made in the test suite starts clean without asking.

    Given a store on `astro_max` with files in it, when a test in the suite makes a run on `astro_max` on that
    instance and names no choice, then the store holds only the empty `log` the run's folder mounts over when PX4
    boots.
    """
    vehicle, store = px4_store

    with Sim(vehicle, scene=sv.SCENE, device="cpu"):
        pass

    assert sorted(p.name for p in store.iterdir()) == ["log"]
