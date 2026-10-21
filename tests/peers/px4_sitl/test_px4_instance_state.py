"""The state a PX4 Software In The Loop (SITL) run keeps: the parameters and the mission PX4 stores outlive the run,
as a real vehicle's outlive a reboot, per airframe and instance, and a run asked to start clean boots with none.

PX4 saves a parameter set at run time, and a mission it accepts, in its working folder. Each test flies runs of the
fixture vehicle through ``nx.Sim`` back to back on one machine, so the second run claims the instance the first
held, and talks to PX4 over the offboard link as a ground station does, with MAVLink alone. It needs docker and a
PX4 tree carrying the airframes the sim flies; it skips without one. Each run that should keep the store asks for
it, since a run in the test suite starts clean unless it asks.
"""

import os
import shutil
import time
from pathlib import Path

import pytest

pytest.importorskip("pymavlink")
pytest.importorskip("newton")
pytest.importorskip("pxr")

# Pin the MAVLink dialect before importing mavutil, as the PX4 controller does: whichever import runs
# first sets it, and the controller's HIL_GPS needs the common dialect's fields.
os.environ.setdefault("MAVLINK20", "1")
os.environ.setdefault("MAVLINK_DIALECT", "common")

from pymavlink import mavutil

import nexus_uav as nx
from nexus_uav._src.peers.px4_sitl import checkout
from nexus_uav.examples._lib.origin import origin_layer
from nexus_uav.examples.controllers.px4.mission import PLAN
from nexus_uav.px4 import OffboardClient, read_plan
from tests.usd import sensor_vehicle as sv

# The tree the run would fly, with no fetch: the checkout the pin names, or the pinned tree once a run fetched it.
_TREE = checkout.tree(fetch_missing=False)
_AIRFRAMES = None if _TREE is None else _TREE / "ROMFS" / "px4fmu_common" / "init.d-posix" / "airframes"
# A mark, not a skip at import, so a run that deselects `px4_sitl` lists none of these tests.
# PX4's ports are the machine's, so the tests that bind or dial them run on one worker, one at a time.
pytestmark = [
    pytest.mark.px4_sitl,
    pytest.mark.xdist_group("px4_ports"),
    pytest.mark.skipif(
        _AIRFRAMES is None or not list(_AIRFRAMES.glob("*_none_astro_max")) or not list(_AIRFRAMES.glob("*_none_iris")),
        reason="needs docker and a PX4 tree on this machine, the pinned checkout or commit, carrying the none_astro_max"
        " and none_iris airframes",
    ),
]

_PARAM = "COM_DISARM_PRFLT"
# The box mission's items: a takeoff, four waypoints and a return to launch.
_BOX_ITEMS = 6
# Where the run's PX4 flies the box mission, so PX4 accepts it.
_LAYER = origin_layer(*read_plan(PLAN).home)


class _Gcs:
    """A ground station on the run's offboard link: what a pilot's QGroundControl asks of PX4, in MAVLink alone."""

    def __init__(self, sim):
        link = sim.ports["offboard"]
        self.instance = link["system_id"] - 1
        self._sim = sim
        self._system = link["system_id"]
        self._mav = mavutil.mavlink_connection(f"udpin:0.0.0.0:{link['port']}", source_system=255, source_component=190)
        self._ask(lambda: None, "HEARTBEAT", lambda m: m.get_srcSystem() == self._system, 30.0)

    def close(self):
        self._mav.close()

    def _ask(self, send, kind, wanted, timeout=10.0):
        """Send with `send` each half second of sim time until a message of `kind` that `wanted` accepts arrives,
        stepping the sim meanwhile, and return it.
        """
        got = []
        sent = [-1.0]

        def answered():
            if self._sim.time - sent[0] >= 0.5:
                self._mav.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
                )
                send()
                sent[0] = self._sim.time
            while (m := self._mav.recv_match(type=kind, blocking=False)) is not None:
                if wanted(m):
                    got.append(m)
            return bool(got)

        self._sim.wait_until(answered, sim_timeout=timeout)
        return got[0]

    def param(self, name):
        """The value PX4 reports for the parameter `name`."""
        return self._ask(
            lambda: self._mav.mav.param_request_read_send(self._system, 1, name.encode(), -1),
            "PARAM_VALUE",
            lambda m: m.param_id == name,
        ).param_value

    def set_param(self, name, value):
        """Set the parameter `name` to `value`, once PX4 reports it, and give PX4 the moment it takes to store it."""
        self._ask(
            lambda: self._mav.mav.param_set_send(
                self._system, 1, name.encode(), value, mavutil.mavlink.MAV_PARAM_TYPE_REAL32
            ),
            "PARAM_VALUE",
            lambda m: m.param_id == name and m.param_value == value,
        )
        self._sim.sleep(2.0)

    def mission_count(self):
        """How many items the mission PX4 holds has."""
        return self._ask(
            lambda: self._mav.mav.mission_request_list_send(self._system, 1), "MISSION_COUNT", lambda m: True
        ).count


def _boot(vehicle, *, clean, then=None):
    """Fly a run of `vehicle` until PX4 answers, asked to start clean or to keep its store, and close it. Returns the
    run's instance, `COM_DISARM_PRFLT` and the mission's item count as PX4 reports them at boot, and what `then`,
    called with the ground station, returned.
    """
    with nx.Sim(vehicle, scene=sv.SCENE, layer=_LAYER, clean=clean) as sim:
        sim.start(timeout=120.0)
        gcs = _Gcs(sim)
        try:
            seen = (gcs.instance, gcs.param(_PARAM), gcs.mission_count(), then(gcs) if then else None)
        finally:
            gcs.close()
    return seen


def _upload_box(vehicle, *, clean):
    """Fly a run of `vehicle` that uploads the box mission until PX4 accepts it, and close it. Returns its instance."""
    with nx.Sim(vehicle, scene=sv.SCENE, layer=_LAYER, clean=clean) as sim:
        sim.start(timeout=120.0)
        with OffboardClient.connect(sim, 30.0) as op:
            op.upload_mission(PLAN)
            sim.wait_until(op.mission_uploaded, sim_timeout=60.0)
            sim.sleep(2.0)  # PX4 stores the mission a moment after it accepts it
        return sim.ports["offboard"]["system_id"] - 1


def _on_airframe(vehicle: str, airframe: str) -> str:
    """The fixture vehicle at `vehicle` with PX4 on `airframe` instead, written beside it."""
    path = Path(vehicle).with_name(f"{airframe}.usda")
    path.write_text(Path(vehicle).read_text().replace('nexus:airframe = "astro_max"', f'nexus:airframe = "{airframe}"'))
    return str(path)


def _written_since(folder: Path, since: float) -> list[str]:
    """The files under ``folder`` last written at or after ``since``, as paths relative to it, sorted."""
    if not folder.is_dir():
        return []
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file() and p.stat().st_mtime >= since)


def test_a_runs_px4_boots_with_a_parameter_an_earlier_run_on_its_airframe_and_instance_set(px4_vehicle):
    """A run's PX4 boots with a parameter an earlier run on the same airframe and instance set.

    Given a real run on `astro_max` that sets `COM_DISARM_PRFLT` to 0 over MAVLink and closes, when a second run on
    `astro_max` starts on the same instance, then its PX4 reports `COM_DISARM_PRFLT` at 0 at boot.
    """
    first, *_ = _boot(px4_vehicle, clean=False, then=lambda gcs: gcs.set_param(_PARAM, 0.0))

    second, value, _, _ = _boot(px4_vehicle, clean=False)

    assert (second, value) == (first, 0.0)


def test_a_runs_px4_boots_with_the_mission_an_earlier_run_on_its_airframe_and_instance_stored(px4_vehicle):
    """A run's PX4 boots with the mission an earlier run on the same airframe and instance stored.

    Given a real run on `astro_max` that uploads `box.plan`, PX4 accepts it, and the run closes, when a second run
    on `astro_max` starts on the same instance, then its PX4 reports a mission of `box.plan`'s item count.
    """
    first = _upload_box(px4_vehicle, clean=False)

    second, _, items, _ = _boot(px4_vehicle, clean=False)

    assert (second, items) == (first, _BOX_ITEMS)


def test_a_run_on_another_airframe_neither_reads_nor_wipes_a_store(px4_vehicle):
    """A run on another airframe neither reads nor wipes a store.

    Given a store on `astro_max`, instance 0, with `COM_DISARM_PRFLT` at 0, when a run on `iris` flies on instance 0
    and then a run on `astro_max` starts there, then `iris` boots with `COM_DISARM_PRFLT` at its airframe value, and
    `astro_max` boots with 0.
    """
    first, *_ = _boot(px4_vehicle, clean=False, then=lambda gcs: gcs.set_param(_PARAM, 0.0))

    iris, on_iris, _, _ = _boot(_on_airframe(px4_vehicle, "iris"), clean=False)
    again, on_astro_max, _, _ = _boot(px4_vehicle, clean=False)

    assert (iris, on_iris, again, on_astro_max) == (first, 10.0, first, 0.0)


def test_a_run_asked_to_start_clean_boots_px4_with_no_parameter_or_mission_an_earlier_run_stored(px4_vehicle):
    """A run asked to start clean boots PX4 with no parameter or mission an earlier run stored.

    Given a store on `astro_max`, instance 0, with `COM_DISARM_PRFLT` at 0 and `box.plan` stored, when a run asked
    to start clean starts on `astro_max` there, then its PX4 reports `COM_DISARM_PRFLT` at 10.0 and a mission of 0
    items.
    """
    _upload_box(px4_vehicle, clean=False)
    first, *_ = _boot(px4_vehicle, clean=False, then=lambda gcs: gcs.set_param(_PARAM, 0.0))

    second, value, items, _ = _boot(px4_vehicle, clean=True)

    assert (second, value, items) == (first, 10.0, 0)


def test_a_runs_px4_keeps_its_store_under_the_px4_peers_cache_folder_and_nothing_in_the_tree(px4_vehicle):
    """A run's PX4 keeps its store under the PX4 peer's cache folder, and nothing in the tree.

    Given a real run on `astro_max` that sets a parameter and closes, then `parameters.bson` lies under
    `~/.cache/nexus/px4_sitl/`, in a folder of that airframe and instance, and the run wrote no file in the tree's
    instance folder or under `~/.cache/nexus/px4-work/`.
    """
    began = time.time()

    instance, *_ = _boot(px4_vehicle, clean=False, then=lambda gcs: gcs.set_param(_PARAM, 0.0))

    cache = Path("~/.cache/nexus").expanduser()
    store = cache / "px4_sitl" / "store" / "astro_max" / str(instance)
    in_tree = _TREE / "build" / "px4_sitl_default" / "rootfs" / str(instance)
    elsewhere = _written_since(in_tree, began) + _written_since(cache / "px4-work", began)
    assert ("parameters.bson" in _written_since(store, began), elsewhere) == (True, [])


def test_a_kept_store_boots_on_a_px4_tree_other_than_the_one_that_filled_it(px4_vehicle, tmp_path):
    """A kept store boots on a PX4 tree other than the one that filled it.

    Given a store on `astro_max` whose links into the PX4 tree point into a tree that no longer exists, when a run
    on `astro_max` starts, then PX4 boots and reports the stored `COM_DISARM_PRFLT` at 0.
    """
    first, *_ = _boot(px4_vehicle, clean=False, then=lambda gcs: gcs.set_param(_PARAM, 0.0))
    store = Path("~/.cache/nexus/px4_sitl/store/astro_max").expanduser() / str(first)
    gone = tmp_path / "gone"
    store.mkdir(parents=True, exist_ok=True)
    for name in ("etc", "test_data"):
        link = store / name
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            shutil.rmtree(link)
        link.symlink_to(gone / name)

    second, value, _, _ = _boot(px4_vehicle, clean=False)

    assert (second, value) == (first, 0.0)
