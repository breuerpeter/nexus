"""The builder starts each peer the vehicle declares, and each one a class requires, through the registry: a
project's own peer starts as a shipped one does, a test sends a peer to its fake with a row of its own
registry, and the receipt records the class each declaration resolved to. A vehicle declares the shipped
PX4 Software In The Loop (SITL) peer, and a project's peer here states a link over the User Datagram
Protocol (UDP).

The docker daemon and the network are the system boundaries: the stand-in daemon of the pin tests records any
container a run starts, and every fetch fails at once on a cut network. Real builds on the Warp CPU backend of
the fixture vehicle in ``tests/usd/sensor_vehicle.py``. Skipped without newton or pxr.
"""

import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

from pathlib import Path
from typing import ClassVar

import nexus_uav._src.build.launch as L
from nexus_uav._src.config import LaunchConfig
from nexus_uav._src.core.registry import Registry
from nexus_uav._src.peers.kit.fake import KitFake
from nexus_uav._src.peers.px4_sitl.fake import Px4Fake
from tests.peers import test_pins as pins
from tests.usd import sensor_vehicle as sv

# PX4's ports are the machine's, so the tests that bind or dial them run on one worker, one at a time.
pytestmark = [pytest.mark.xdist_group("px4_ports"), pytest.mark.usefixtures("warp_cpu")]

# The stand-in docker daemon of the pin tests, with a home folder in the test's folder and the network cut.
daemon = pins.daemon


def _counting() -> type:
    """A fake PX4 peer class that keeps the instance of each fake that started in the class's `starts`.

    Each call makes a class of its own, for one run.
    """

    class Counting(Px4Fake):
        starts: ClassVar[list] = []

        def start(self) -> None:
            type(self).starts.append(self.instance)
            super().start()

    return Counting


class _RadarSensor(sv.StandInSensor):
    """A stand-in sensor whose class requires a peer named `radar`."""

    requires = ("radar",)


def _built(path: str, registry: Registry, during=lambda loop: None) -> dict:
    """Build the vehicle at `path` in the fixture scene with `registry`, call `during` on the built run, and close
    it. Returns what `during` returned, the run's receipt and its port map, or the error the build raised.
    """
    launch = LaunchConfig.from_dict({"vehicle": path, "scene": sv.SCENE, "runtime": {"device": "cpu"}})
    try:
        loop = L.build_from_launch(launch, preroll_timeout=10.0, components=registry)
    except Exception as exc:
        return {"during": None, "receipt": {}, "ports": {}, "error": exc}
    try:
        during_value = during(loop)
        ports = dict(loop.ports)
    finally:
        loop.close()
    return {"during": during_value, "receipt": loop.settings, "ports": ports, "error": None}


def _resolved(receipt: dict, schema: str) -> tuple[str, str] | None:
    """The prim that applies `schema` and the name of the class it resolved to, as the receipt records them."""
    for entry in receipt.get("declarations", []):
        if entry["schema"] == schema:
            return entry["prim"], entry["class"].rsplit(":", 1)[-1].rsplit(".", 1)[-1]
    return None


def _steps(ticks: int):
    """A `during` that steps the run `ticks` ticks and returns how many it ran."""

    def step(loop) -> int:
        n = 0
        while n < ticks and loop.step():
            n += 1
        return n

    return step


def test_a_peer_a_project_declares_starts_with_its_schemas_and_the_runs_values_and_stops_at_the_runs_end(
    daemon, tmp_path
):
    """A peer a project declares on the vehicle starts with its schema's values and the run's values, and stops
    at the run's end.

    Given the fixture vehicle that declares the PX4 SITL peer and, on its scope `Companion`, the stand-in peer
    schema with the greeting `hello`, and a `Registry` that maps both to stand-ins, when the run builds and
    closes, then the build gave the stand-in peer `hello`, its instance and the run's folder, the peer started
    before the first tick, and it stopped at close.
    """
    peer = sv.radio()
    path = sv.vehicle(tmp_path, root=sv.companion("hello"), px4=True)
    registry = sv.components(NexusPx4SitlAPI=Px4Fake, StandInPeerAPI=peer)

    outcome = _built(path, registry, during=lambda loop: list(peer.events))

    run = peer.built[0] if peer.built else {}
    seen = (
        run.get("greeting"),
        isinstance(run.get("instance"), int),
        Path(run.get("log_dir") or tmp_path / "none").is_dir(),
        outcome["during"],
        peer.events,
    )
    assert seen == ("hello", True, True, ["start"], ["start", "stop"]), outcome["error"]


def test_a_test_sends_each_shipped_peer_to_its_fake_through_the_registry_alone(daemon, tmp_path):
    """A test sends each shipped peer to its fake through the registry alone.

    Given the fixture vehicle with a camera that declares the PX4 SITL peer, and a `Registry` that maps the PX4
    SITL peer's schema to `Px4Fake` and `kit` to `KitFake`, when the run steps 5 ticks, then it flies all 5
    against both fakes and the stand-in docker daemon ran no container.
    """
    path = sv.vehicle(tmp_path, pins.CAMERA, px4=True)
    registry = Registry({"NexusPx4SitlAPI": Px4Fake, "kit": KitFake})

    outcome = _built(path, registry, during=_steps(5))

    assert (outcome["during"], daemon.runs) == (5, []), outcome["error"]


def test_the_receipt_records_the_class_each_declaration_resolved_to(daemon, tmp_path):
    """The receipt records the class each declaration resolved to.

    Given a run whose `Registry` maps the PX4 SITL peer's schema to `Px4Fake` and the PX4 schema to the stand-in
    controller, when the run builds, then the receipt names `Px4Fake` for the PX4 SITL peer's prim and the
    stand-in controller for the controller's; and given a run with no row for the PX4 SITL peer, on the stand-in
    daemon with the shipped pin's tree on this machine, it names `Px4Sitl`.
    """
    ref = pins._shipped("px4_sitl").read_text().splitlines()[0].strip()
    tree = tmp_path / "home" / ".cache" / "nexus" / "px4_sitl" / ref.split("@", 1)[1]
    tree.mkdir(parents=True)
    (tree / "Makefile").write_text("px4_sitl:\n\ttrue\n")
    path = sv.vehicle(tmp_path, px4=True)

    faked = _built(path, sv.components(NexusPx4SitlAPI=Px4Fake))
    real = _built(path, sv.components())

    seen = (
        _resolved(faked["receipt"], "NexusPx4SitlAPI"),
        _resolved(faked["receipt"], "NexusPx4API_1"),
        _resolved(real["receipt"], "NexusPx4SitlAPI"),
    )
    expected = (
        ("/vehicle/Px4Sitl", "Px4Fake"),
        ("/vehicle/Controller", "Controller"),
        ("/vehicle/Px4Sitl", "Px4Sitl"),
    )
    assert seen == expected, (faked["error"], real["error"])


def test_a_link_a_projects_peer_states_as_leaving_the_run_is_in_the_runs_port_map(daemon, tmp_path):
    """A link a project's peer states as leaving the run is in the run's port map.

    Given the fixture vehicle that declares the stand-in peer, which states one link that leaves the run,
    `radio`, on UDP port 50000 plus its instance, when the run builds, then the run's port map names `radio`
    with that port.
    """
    peer = sv.radio()
    path = sv.vehicle(tmp_path, root=sv.companion(), px4=True)

    outcome = _built(path, sv.components(NexusPx4SitlAPI=Px4Fake, StandInPeerAPI=peer))

    instance = peer.built[0]["instance"] if peer.built else None
    expected = {"protocol": "udp", "port": 50000 + instance} if instance is not None else "a radio peer that ran"
    assert outcome["ports"].get("radio") == expected, outcome["error"]


def test_a_class_that_requires_a_peer_the_registry_maps_nothing_to_fails_before_any_peer_starts(daemon, tmp_path):
    """A class that requires a peer the registry maps nothing to fails the build before any peer starts.

    Given the fixture vehicle that declares the PX4 SITL peer and a stand-in sensor whose class requires a peer
    named `radar`, and a `Registry` that maps nothing to `radar`, when the run builds, then it raises naming the class
    and `radar`, and no peer started.
    """
    fake = _counting()
    path = sv.vehicle(tmp_path, sv.prim("Radar", "StandInSensorAPI"), px4=True)

    outcome = _built(path, sv.components(NexusPx4SitlAPI=fake, StandInSensorAPI=_RadarSensor))

    message = str(outcome["error"] or "")
    seen = ("_RadarSensor" in message, "radar" in message, fake.starts, daemon.started())
    assert seen == (True, True, [], []), outcome["error"]


def test_a_peer_schema_no_class_claims_fails_before_any_peer_starts(daemon, tmp_path):
    """A peer schema no class claims fails the build before any peer starts.

    Given the fixture vehicle that declares the PX4 SITL peer and, on its scope `Companion`, the stand-in peer
    schema, and a `Registry` that maps nothing to that schema, when the run builds, then it raises naming the
    prim and the schema and saying no class claims it, and no peer started.
    """
    fake = _counting()
    path = sv.vehicle(tmp_path, root=sv.companion(), px4=True)

    outcome = _built(path, sv.components(NexusPx4SitlAPI=fake))

    message = str(outcome["error"] or "")
    named = [s in message for s in ("/vehicle/Companion", "StandInPeerAPI", "no class claims")]
    assert (named, fake.starts, daemon.started()) == ([True] * 3, [], []), outcome["error"]
