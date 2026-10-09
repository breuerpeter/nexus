"""PX4 still receives what it received before each sensor's output became a signal: every
Hardware In The Loop (HIL) message the fake PX4 peer received from ``main``, it receives again at the same time,
equal in every field but ``fields_updated``. A sensor sets its bits there only on a tick that brings it a new
sample, and the Global Positioning System (GPS) receiver here declares no rate, so ``HIL_GPS`` goes out every
tick, where ``main`` sent it at a 10 Hz sub-rate.

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
from tests.usd import sensor_vehicle as sv

# PX4's ports are the machine's, so the tests that bind or dial them run on one worker, one at a time.
pytestmark = [pytest.mark.usefixtures("warp_cpu"), pytest.mark.xdist_group("px4_ports")]

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


def _received(tmp_path: Path, body: str) -> list[dict]:
    """What the fake receives in a run of the fixture with the prims `body` under its base body: each message of
    the three kinds the test compares, as its fields, in the order it arrived, from the first 250 exchanges.

    A run's messages of one exchange all reach the fake before the next exchange's `HIL_SENSOR`, which it
    answers before the run steps on, so the run steps one exchange more than the test compares.
    """
    loop = sv.build(
        sv.vehicle(tmp_path, body, px4=True),
        seed=SEED,
        components=default_registry(),
        peers={"px4_sitl": sv.KeepingFake},
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


def _unmasked(msg: dict) -> dict:
    """A message's fields, but the bits of `fields_updated`."""
    return {field: value for field, value in msg.items() if field != "fields_updated"}


def test_px4_receives_every_hil_message_main_sent_at_its_time_field_for_field(tmp_path):
    """PX4 receives every message it received from `main`, at the same time, field for field, but the bits each sensor sets.

    Given the fake PX4 peer, one seed and the fixture vehicle with an Inertial Measurement Unit (IMU), a
    magnetometer, a barometer and a GPS receiver, none of which declares a rate, when the run steps 250 ticks,
    then for every `HIL_SENSOR`, `HIL_GPS` and `HIL_STATE_QUATERNION` the fake received from `main`, it receives
    one of the same kind at the same `time_usec`, equal in every field but `fields_updated`; once more with only
    the IMU and the GPS, and once with only the IMU.
    """
    main = json.loads(gzip.decompress(MAIN.read_bytes()))
    missing = {}
    for name, body in SUITES.items():
        folder = tmp_path / name
        folder.mkdir()
        received = {(msg["mavpackettype"], msg["time_usec"]): _unmasked(msg) for msg in _received(folder, body)}
        missing[name] = [
            msg for msg in main[name] if received.get((msg["mavpackettype"], msg["time_usec"])) != _unmasked(msg)
        ]

    assert missing == {name: [] for name in SUITES}
