"""PX4 receives what it received before each sensor's output became a signal: every Hardware In The Loop (HIL)
message the fake PX4 peer receives equals, field for field, what it received from ``main``.

Real builds on the Warp CPU backend of the fixture vehicle in ``tests/usd/sensor_vehicle.py``, flown by the PX4
controller against the fake PX4 peer, which keeps every message it receives. Each sensor is quiet: the passes
before PX4 attaches each sample the sensors, and their number depends on when the fake dials in, so a noisy
sample would differ from run to run. Skipped without newton or pxr.
"""

import gzip
import json
from pathlib import Path

import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

from nexus_sim._src.core.registry import default_registry
from nexus_sim._src.peers.px4_sitl.fake import Px4Fake
from tests.usd import sensor_vehicle as sv

pytestmark = pytest.mark.usefixtures("warp_cpu")

# What the fake received from main at b4443f7, before each sensor's output became a signal: for each set of
# sensors, `_received` of a run of the fixture with it, as JSON, compressed. Never re-record it after the change.
MAIN = Path(__file__).with_name("hil_main.json.gz")
KINDS = ("HIL_SENSOR", "HIL_GPS", "HIL_STATE_QUATERNION")
EXCHANGES = 250
SEED = 7

IMU = sv.prim("Imu0", "NexusImuAPI", "float nexus:accNoise = 0\nfloat nexus:gyroNoise = 0")
MAG = sv.prim("Mag0", "NexusMagAPI", "float3 nexus:noise = (0, 0, 0)")
BARO = sv.prim("Baro0", "NexusBaroAPI", "float nexus:noise = 0")
GPS = sv.prim("Gps0", "NexusGpsAPI")
# The sets of sensors a run flies, each under the name its messages sit at in the recording from main.
SUITES = {"imu_mag_baro_gps": IMU + MAG + BARO + GPS, "imu_gps": IMU + GPS, "imu": IMU}


class _Every(dict):
    """The fake's last message of each kind, which also keeps, in order, every message the fake sets in it."""

    def __init__(self):
        super().__init__()
        self.every: list = []

    def __setitem__(self, kind, msg):
        self.every.append(msg)
        super().__setitem__(kind, msg)


class _KeepingFake(Px4Fake):
    """The fake PX4 peer, which keeps every message it receives, in order, beside the last of each kind."""

    def __init__(self, **run):
        super().__init__(**run)
        self.last = _Every()


def _received(tmp_path: Path, body: str) -> list[dict]:
    """What the fake receives in a run of the fixture with the prims `body` under its base body: each message of
    the three kinds the test compares, as its fields, in the order it arrived, from the first 250 exchanges.

    A run's messages of one exchange all reach the fake before the next exchange's `HIL_SENSOR`, which it
    answers before the run steps on, so the run steps one exchange more than the test compares.
    """
    loop = sv.build(
        sv.vehicle(tmp_path, body, px4=True), seed=SEED, components=default_registry(), peers={"px4_sitl": _KeepingFake}
    )
    fake = loop.peers[0]
    sv.steps(loop, EXCHANGES + 1)
    kept, sensors = [], 0
    for msg in fake.last.every:
        sensors += msg.get_type() == "HIL_SENSOR"
        if sensors > EXCHANGES:
            break
        if msg.get_type() in KINDS:
            kept.append(msg.to_dict())
    return kept


def test_px4_receives_every_hil_message_field_for_field_as_it_did_from_main(tmp_path):
    """PX4 receives what it receives today.

    Given the fake PX4 peer, one seed and the fixture vehicle with an Inertial Measurement Unit (IMU), a
    magnetometer, a barometer and a Global Positioning System (GPS) receiver, when the run steps 250 ticks, then
    every `HIL_SENSOR`, `HIL_GPS` and `HIL_STATE_QUATERNION` the fake receives equals, field for field, what it
    received from `main`; once more with only the IMU and the GPS, and once with only the IMU.
    """
    main = json.loads(gzip.decompress(MAIN.read_bytes()))
    received = {}
    for name, body in SUITES.items():
        folder = tmp_path / name
        folder.mkdir()
        received[name] = _received(folder, body)

    assert received == main
