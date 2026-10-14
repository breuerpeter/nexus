"""What a script reads when PX4 leaves a run with no verdict, #321: its own error lines in the run's ``SetupError``, and
``OffboardClient.connect``, which waits for PX4's heartbeat in sim time.

The tests marked ``px4_sitl`` fly through ``nx.Sim``, which builds and launches PX4 from the tree on this
machine, so they need docker and a PX4 tree carrying the airframe the sim flies; they skip without one. The
heartbeat timeout runs on the fixture vehicle on the Warp CPU backend, with the PX4 Software In The Loop (SITL)
peer sent to a stand-in of its fake that names an offboard port nothing answers on.
"""

import os
import socket

import pytest

pytest.importorskip("pymavlink")
pytest.importorskip("newton")
pytest.importorskip("pxr")

# Pin the MAVLink dialect before importing mavutil, as the PX4 controller does.
os.environ.setdefault("MAVLINK20", "1")
os.environ.setdefault("MAVLINK_DIALECT", "common")

import nexus_uav as nx
from nexus_uav._src.core.ports import PortMap
from nexus_uav._src.peers.px4_sitl import checkout
from nexus_uav._src.peers.px4_sitl.fake import Px4Fake
from nexus_uav.px4 import OffboardClient
from tests.usd import sensor_vehicle as sv

# PX4's ports are the machine's, so the tests that bind or dial them run on one worker, one at a time.
pytestmark = pytest.mark.xdist_group("px4_ports")

# The tree the run would fly, with no fetch: the checkout the pin names, or the pinned tree once a run fetched it.
_TREE = checkout.tree(fetch_missing=False)
# A skip mark, not a skip at import, so a run that deselects `px4_sitl` lists none of these tests.
_NEEDS_TREE = pytest.mark.skipif(
    _TREE is None
    or not list((_TREE / "ROMFS" / "px4fmu_common" / "init.d-posix" / "airframes").glob("*_none_astro_max")),
    reason="needs docker and a PX4 tree on this machine, the pinned checkout or commit, carrying the none_astro_max airframe",
)


def _raised(call) -> BaseException | None:
    """The exception `call` raises, or `None` when it returns."""
    try:
        call()
    except Exception as exc:  # the test reads which error the call raised
        return exc
    return None


def _caught_as_setup_error(exc: BaseException | None) -> bool:
    """Whether `except nexus_uav.SetupError` catches `exc`."""
    return exc is not None and isinstance(exc, getattr(nx, "SetupError", ()))


@pytest.mark.px4_sitl
@_NEEDS_TREE
@pytest.mark.usefixtures("warp_cpu")
def test_a_px4_tree_without_the_vehicles_airframe_fails_the_run_with_px4s_own_error(tmp_path):
    """A PX4 tree without the vehicle's airframe fails the run with PX4's own error in the message.

    Given the fixture vehicle on airframe `no_such_airframe`, which the pinned tree lacks, when `Sim.start(timeout=60)`
    runs, then it raises `SetupError` whose message holds PX4's `Unknown model` line.
    """
    path = sv.vehicle(tmp_path, px4=True)
    with open(path) as f:
        text = f.read().replace('nexus:airframe = "astro_max"', 'nexus:airframe = "no_such_airframe"')
    with open(path, "w") as f:
        f.write(text)

    with nx.Sim(path, scene=sv.SCENE, device="cpu") as sim:
        error = _raised(lambda: sim.start(timeout=60.0))

    assert (_caught_as_setup_error(error), "Unknown model" in str(error)) == (True, True), error


@pytest.mark.px4_sitl
@_NEEDS_TREE
def test_offboard_connect_returns_a_client_that_has_px4s_heartbeat():
    """`OffboardClient.connect(sim, timeout)` returns a client that has PX4's heartbeat.

    Given `astro_max_base` started on PX4, when `OffboardClient.connect(sim, 10.0)` returns, then the client has
    the heartbeat, and a `takeoff(2.0)` it sends reaches its target within 60 s of sim time.
    """
    with nx.Sim("astro_max_base", scene="empty") as sim:
        sim.start(timeout=120.0)
        clients = []
        error = _raised(lambda: clients.append(OffboardClient.connect(sim, 10.0)))
        connected = reached = False
        if clients:
            op = clients[0]
            try:
                connected = op.connected
                op.takeoff(2.0)
                reached = _raised(lambda: sim.wait_until(op.at_target, sim_timeout=60.0)) is None
            finally:
                op.close()

    assert (error, connected, reached) == (None, True, True), error


def _free_udp_port() -> int:
    """A User Datagram Protocol (UDP) port nothing on this machine listens on."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _NoOffboard(Px4Fake):
    """The fake PX4, which also names an offboard port in the run's port map, where nothing answers."""

    port = _free_udp_port()

    def ports(self) -> PortMap:
        return PortMap({"offboard": {"protocol": "udp", "port": self.port, "system_id": self.instance + 1}})


@pytest.mark.usefixtures("warp_cpu")
def test_offboard_connect_raises_the_setup_error_when_no_heartbeat_comes_in_time(tmp_path):
    """`OffboardClient.connect` raises `SetupError` when no heartbeat comes within `timeout` sim seconds.

    Given a run of the fixture vehicle whose fake PX4 names an offboard port that nothing answers on, when
    `OffboardClient.connect(sim, 1.0)` runs, then it raises `SetupError` naming the port, with the sim at least
    1.0 s later.
    """
    vehicle = sv.vehicle(tmp_path, px4=True)
    registry = nx.Registry({"NexusPx4SitlAPI": _NoOffboard})
    with nx.Sim(vehicle, scene=sv.SCENE, device="cpu", registry=registry) as sim:
        sim.start(timeout=10.0)
        t0 = sim.time
        error = _raised(lambda: OffboardClient.connect(sim, 1.0).close())
        waited = sim.time - t0

    named = str(_NoOffboard.port) in str(error)
    assert (_caught_as_setup_error(error), named, waited >= 1.0) == (True, True, True), (error, waited)
