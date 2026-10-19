"""The state a PX4 Software In The Loop (SITL) run leaves: the parameters and the mission it stores land in the
instance's work folder, which the next start empties, and nothing lands in the tree.

PX4 saves a parameter set at run time, and a mission it accepts, in its working folder. The test flies one run of
the fixture vehicle through ``nx.Sim`` that sets ``COM_DISARM_PRFLT`` and uploads a mission over the offboard
link, so it needs docker and a PX4 tree carrying the airframe the sim flies; it skips without one. That a start
empties the folder, whether the run that filled it ended or died before its teardown, the stand-in daemon's test
in ``test_px4_sitl.py`` proves with no boot.
"""

import time
from pathlib import Path

import pytest

pytest.importorskip("pymavlink")
pytest.importorskip("newton")
pytest.importorskip("pxr")

import nexus_uav as nx
from nexus_uav._src.peers.px4_sitl import checkout
from nexus_uav.examples._lib.origin import origin_layer
from nexus_uav.examples.controllers.px4.mission import PLAN
from nexus_uav.px4 import OffboardClient, read_plan
from tests.usd import sensor_vehicle as sv

# The tree the run would fly, with no fetch: the checkout the pin names, or the pinned tree once a run fetched it.
_TREE = checkout.tree(fetch_missing=False)
# A mark, not a skip at import, so a run that deselects `px4_sitl` lists none of these tests.
# PX4's ports are the machine's, so the tests that bind or dial them run on one worker, one at a time.
pytestmark = [
    pytest.mark.px4_sitl,
    pytest.mark.xdist_group("px4_ports"),
    pytest.mark.skipif(
        _TREE is None
        or not list((_TREE / "ROMFS" / "px4fmu_common" / "init.d-posix" / "airframes").glob("*_none_astro_max")),
        reason="needs docker and a PX4 tree on this machine, the pinned checkout or commit, carrying the none_astro_max airframe",
    ),
]

_PARAM = "COM_DISARM_PRFLT"


def _written_since(folder: Path, since: float) -> list[str]:
    """The files under ``folder`` last written at or after ``since``, as paths relative to it, sorted."""
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file() and p.stat().st_mtime >= since)


def test_a_runs_px4_stores_its_parameters_and_mission_in_the_instances_work_folder_and_nothing_in_the_tree(px4_vehicle):
    """A run's PX4 stores the parameters and the mission it sets in the instance's work folder, and nothing in the tree.

    Given a real run that sets `COM_DISARM_PRFLT` to 0 and uploads `box.plan` over the offboard link, PX4 accepts
    it, and the run closes, then the instance's work folder, `~/.cache/nexus/px4-work/<instance>`, holds
    `parameters.bson` and `dataman` written during the run, and the tree's folder of the instance holds no file
    written during it.
    """
    began = time.time()
    with nx.Sim(px4_vehicle, scene=sv.SCENE, layer=origin_layer(*read_plan(PLAN).home)) as sim:
        sim.start(timeout=120.0)
        with OffboardClient.connect(sim, 30.0) as op:
            op.upload_mission(PLAN)
            sim.wait_until(op.mission_uploaded, sim_timeout=60.0)
            for _ in range(3):  # fire and forget, so send it more than once
                op.param_set(_PARAM, 0.0)
                sim.sleep(1.0)
            sim.sleep(2.0)  # PX4 stores a change a moment after it
        instance = sim.ports["offboard"]["system_id"] - 1

    work = Path("~/.cache/nexus/px4-work").expanduser() / str(instance)
    in_tree = _TREE / "build" / "px4_sitl_default" / "rootfs" / str(instance)
    stored = _written_since(work, began)
    assert ("parameters.bson" in stored, "dataman" in stored, _written_since(in_tree, began)) == (True, True, [])
