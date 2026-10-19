"""The serial link the run owns between a declared peer and the PX4 Software In The Loop (SITL) peer: the
run makes it before any peer starts, hands each peer a path, and closes it at the run's end. A vehicle that
declares no such peer flies PX4 on its User Datagram Protocol (UDP) links alone.

The docker daemon and the network are the system boundaries: the stand-in daemon of the pin tests records any
container a run starts, and every fetch fails at once on a cut network. Real builds on the Warp CPU backend of
the fixture vehicle in ``tests/usd/sensor_vehicle.py``, with PX4 on its fake. Skipped without newton or pxr.
"""

import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import os
import threading
from typing import ClassVar

import nexus_uav._src.build.launch as L
from nexus_uav._src.config import LaunchConfig
from nexus_uav._src.core.registry import Registry
from nexus_uav._src.peers.px4_sitl.fake import Px4Fake
from tests.peers import test_pins as pins
from tests.usd import sensor_vehicle as sv

# PX4's ports are the machine's, so the tests that bind or dial them run on one worker, one at a time.
pytestmark = [pytest.mark.xdist_group("px4_ports"), pytest.mark.usefixtures("warp_cpu")]

# The stand-in docker daemon of the pin tests, with a home folder in the test's folder and the network cut.
daemon = pins.daemon


def _open_tty(path) -> bool:
    """Whether `path` names a device this process can open as a terminal, as a peer opens its end of a serial link."""
    if not path:
        return False
    try:
        fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    except OSError:
        return False
    try:
        return os.isatty(fd)
    finally:
        os.close(fd)


def _companion() -> type:
    """A stand-in class of a project's companion peer, which links to the PX4 SITL peer over a serial line.

    The class keeps, in `ttys`, whether the path the build handed it for that link was an open terminal when its
    `start` ran. Each call makes a class of its own, for one run.
    """

    class Companion(sv.radio()):
        serial = ("px4_sitl",)
        ttys: ClassVar[list] = []

        def start(self) -> None:
            type(self).ttys.append(_open_tty(self.path()))
            super().start()

        def path(self):
            return dict(self.run.get("serial") or {}).get("px4_sitl")

    return Companion


def _fake() -> type:
    """A fake PX4 peer class that keeps the serial paths the build handed each fake, by the other peer's name, in
    `handed`, and in `ttys` whether the path to `radio` was an open terminal when its `start` ran. Each call makes
    a class of its own, for one run.
    """

    class Fake(Px4Fake):
        handed: ClassVar[list] = []
        ttys: ClassVar[list] = []

        def __init__(self, **run) -> None:
            super().__init__(**run)
            self.serial = dict(run.get("serial") or {})
            type(self).handed.append(self.serial)

        def start(self) -> None:
            type(self).ttys.append(_open_tty(self.serial.get("radio")))
            super().start()

    return Fake


def _built(path: str, registry: Registry, during=lambda loop: None) -> dict:
    """Build the vehicle at `path` in the fixture scene with `registry`, call `during` on the built run, and close
    it. Returns what `during` returned, or the error the build raised.
    """
    launch = LaunchConfig.from_dict({"vehicle": path, "scene": sv.SCENE, "runtime": {"device": "cpu"}})
    try:
        loop = L.build_from_launch(launch, preroll_timeout=10.0, components=registry)
    except Exception as exc:
        return {"during": None, "error": exc}
    try:
        return {"during": during(loop), "error": None}
    finally:
        loop.close()


def _steps(ticks: int):
    """A `during` that steps the run `ticks` ticks and returns how many it ran."""

    def step(loop) -> int:
        n = 0
        while n < ticks and loop.step():
            n += 1
        return n

    return step


def test_both_ends_of_the_serial_link_exist_before_either_peer_starts(daemon, tmp_path):
    """Both ends of the serial link exist before either peer starts.

    Given a stand-in declared peer that links to the PX4 SITL peer over a serial line, and PX4 on its fake, when the
    run builds, then each peer's `start` found the path the build handed it, by the other peer's name, an open
    terminal, whichever started first.
    """
    companion, fake = _companion(), _fake()
    path = sv.vehicle(tmp_path, root=sv.companion(), px4=True)

    outcome = _built(path, sv.components(NexusPx4SitlAPI=fake, StandInPeerAPI=companion))

    assert (companion.ttys, fake.ttys) == ([True], [True]), (outcome["error"], companion.built, fake.handed)


def test_a_runs_end_closes_the_serial_link(daemon, tmp_path):
    """A run's end closes the serial link.

    Given a run that joined a stand-in declared peer and the PX4 fake over a serial line, when the run closes, then
    neither path the build handed out is an open terminal, and no thread the run started survives.
    """
    companion, fake = _companion(), _fake()
    path = sv.vehicle(tmp_path, root=sv.companion(), px4=True)
    before = set(threading.enumerate())

    outcome = _built(path, sv.components(NexusPx4SitlAPI=fake, StandInPeerAPI=companion))

    ends = [companion.built[0].get("serial", {}).get("px4_sitl") if companion.built else None]
    ends.append(fake.handed[0].get("radio") if fake.handed else None)
    survivors = [t.name for t in threading.enumerate() if t not in before and t.is_alive()]
    seen = ([bool(end) for end in ends], [_open_tty(end) for end in ends], survivors)
    assert seen == ([True, True], [False, False], []), (outcome["error"], ends)


def test_a_vehicle_that_declares_no_serial_peer_flies_px4_on_its_udp_links_alone(daemon, tmp_path):
    """A vehicle that declares no such peer flies PX4 on its UDP links alone.

    Given the fixture vehicle that declares the PX4 SITL peer and no peer that links to it over a serial line, with
    PX4 on its fake, when the run steps 5 ticks, then the build handed the fake no serial path and the run flew all 5.
    """
    fake = _fake()
    path = sv.vehicle(tmp_path, px4=True)

    outcome = _built(path, sv.components(NexusPx4SitlAPI=fake), during=_steps(5))

    assert (fake.handed, outcome["during"]) == ([{}], 5), outcome["error"]
