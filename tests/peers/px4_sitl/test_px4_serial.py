"""The serial link between the PX4 Software In The Loop (SITL) peer and a declared peer, from the far end: PX4
opens a MAVLink instance on its end of the link the run owns, and PX4's fake answers heartbeats on it as it answers
the Hardware In The Loop (HIL) link.

The real test, marked ``px4_sitl``, flies the fixture vehicle through ``nx.Sim``, which builds and launches PX4
from the tree on this machine, so it needs docker and a PX4 tree carrying the airframe the sim flies; it skips
without one. The fake's test builds the fixture vehicle on the Warp CPU backend with the stand-in docker daemon
of the pin tests and the network cut.
"""

import os
import select
import threading
import time
import tty

import pytest

pytest.importorskip("pymavlink")
pytest.importorskip("newton")
pytest.importorskip("pxr")

# Pin the MAVLink dialect before importing mavutil, as the PX4 controller does.
os.environ.setdefault("MAVLINK20", "1")
os.environ.setdefault("MAVLINK_DIALECT", "common")

from pymavlink import mavutil

import nexus_uav as nx
import nexus_uav._src.build.launch as L
from nexus_uav._src.config import LaunchConfig
from nexus_uav._src.peers.px4_sitl import checkout
from nexus_uav._src.peers.px4_sitl.fake import Px4Fake
from tests.peers import test_pins as pins
from tests.usd import sensor_vehicle as sv

# PX4's ports are the machine's, so the tests that bind or dial them run on one worker, one at a time.
pytestmark = pytest.mark.xdist_group("px4_ports")

# The tree the run would fly, with no fetch: $PX4_DIR, or the pinned tree once a run fetched it.
_TREE = checkout.tree(fetch_missing=False)
# A skip mark, not a skip at import, so a run that deselects `px4_sitl` lists none of these tests.
_NEEDS_TREE = pytest.mark.skipif(
    _TREE is None
    or not list((_TREE / "ROMFS" / "px4fmu_common" / "init.d-posix" / "airframes").glob("*_none_astro_max")),
    reason="needs docker and a PX4 tree on this machine, $PX4_DIR or the fetched pin, carrying the none_astro_max airframe",
)

# The stand-in docker daemon of the pin tests, with a home folder in the test's folder and the network cut.
daemon = pins.daemon


def _end(companion: type):
    """The path of the companion's end of its serial line `autopilot`, as the build handed it, or `None`."""
    return dict(companion.built[0].get("serial") or {}).get("autopilot") if companion.built else None


class _Companion:
    """The far end of the serial link: a MAVLink peer on the companion's device, as its router would be. It
    heartbeats as an onboard controller when asked, and its pump thread keeps the first heartbeat PX4 or its
    fake sends back.
    """

    def __init__(self, path: str):
        self.fd = os.open(path, os.O_RDWR | os.O_NOCTTY)
        tty.setraw(self.fd)
        self.heartbeat = None
        self._mav = mavutil.mavlink.MAVLink(
            os.fdopen(self.fd, "wb", buffering=0, closefd=False), srcSystem=1, srcComponent=191
        )
        self._parser = mavutil.mavlink.MAVLink(None)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._pump, name="companion-end", daemon=True)
        self._thread.start()

    def send_heartbeat(self) -> None:
        """One heartbeat of an onboard controller, as a companion's router sends it."""
        self._mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
        )

    def wait(self, seconds: float) -> bool:
        """Whether a heartbeat came back within `seconds` of wall time."""
        deadline = time.monotonic() + seconds
        while self.heartbeat is None and time.monotonic() < deadline:
            time.sleep(0.05)
        return self.heartbeat is not None

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        os.close(self.fd)

    def _pump(self) -> None:
        while not self._stop.is_set():
            if not select.select([self.fd], [], [], 0.1)[0]:
                continue
            try:
                data = os.read(self.fd, 4096)
            except OSError:
                return
            for msg in self._parser.parse_buffer(data) or []:
                if msg.get_type() == "HEARTBEAT" and self.heartbeat is None:
                    self.heartbeat = msg


@pytest.mark.usefixtures("warp_cpu")
def test_px4s_fake_answers_heartbeats_on_the_serial_link_as_on_the_hil_link(daemon, tmp_path):
    """PX4's fake answers heartbeats on the serial link as on the HIL link.

    Given the PX4 fake and a stand-in declared peer whose prim wires its serial line to the fake's, when the stand-in
    writes a MAVLink heartbeat to its end, then a heartbeat from the fake comes back on it within 2 s.
    """
    companion = sv.radio()
    path = sv.vehicle(tmp_path, root=sv.companion(autopilot=sv.PX4_PRIM), px4=True)
    launch = LaunchConfig.from_dict({"vehicle": path, "scene": sv.SCENE, "runtime": {"device": "cpu"}})
    registry = sv.components(NexusPx4SitlAPI=Px4Fake, StandInPeerAPI=companion)

    loop = L.build_from_launch(launch, preroll_timeout=10.0, components=registry)
    try:
        answered = False
        if _end(companion):
            end = _Companion(_end(companion))
            try:
                end.send_heartbeat()
                answered = end.wait(2.0)
            finally:
                end.close()
    finally:
        loop.close()

    assert answered, companion.built


@pytest.mark.px4_sitl
@_NEEDS_TREE
def test_the_pinned_px4_opens_a_mavlink_instance_on_its_end_of_the_serial_link(tmp_path):
    """The pinned PX4 opens a MAVLink instance on its end of the serial link.

    Given the fixture vehicle that declares a stand-in peer whose prim wires its serial line to the PX4 SITL peer's
    prim, when the run starts the pinned PX4, then a MAVLink heartbeat from PX4 arrives on the stand-in's end within
    10 s.
    """
    companion = sv.radio()
    sensors = (
        sv.prim("Imu", "NexusImuAPI")
        + sv.prim("Mag", "NexusMagAPI", "float nexus:rate = 100")
        + sv.prim("Baro", "NexusBaroAPI", "float nexus:rate = 50")
        + sv.prim("Gps", "NexusGpsAPI", "float nexus:rate = 30")
    )
    path = sv.vehicle(tmp_path, sensors, root=sv.companion(autopilot=sv.PX4_PRIM), px4=True)

    arrived = False
    with nx.Sim(path, scene=sv.SCENE, registry=nx.Registry({"StandInPeerAPI": companion})) as sim:
        sim.start(timeout=120.0)
        if _end(companion):
            end = _Companion(_end(companion))
            try:
                # PX4 keeps lockstep time, so the sim steps while the far end waits for its heartbeat.
                sim.wait_until(lambda: end.heartbeat is not None, sim_timeout=10.0)
            except Exception:  # no heartbeat in time: the assertion below says so
                pass
            finally:
                arrived = end.heartbeat is not None
                end.close()

    assert arrived, companion.built
