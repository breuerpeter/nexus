"""What PX4 receives over the Hardware In The Loop (HIL) link follows each sensor's declared rate.

Real builds of the fixture vehicle in ``tests/usd/sensor_vehicle.py`` on the Warp CPU backend and a 250 Hz tick,
flown by the PX4 controller with the PX4 Software In The Loop (SITL) peer sent to the fake that keeps every
message it receives, so a test counts what PX4 receives in a window of ticks. Skipped without newton or pxr.
"""

import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

from nexus_sim._src.core.registry import default_registry
from tests.usd import sensor_vehicle as sv

# PX4's ports are the machine's, so the tests that bind or dial them run on one worker, one at a time.
pytestmark = [pytest.mark.usefixtures("warp_cpu"), pytest.mark.xdist_group("px4_ports")]

# The bits of `fields_updated` that PX4's `simulator_mavlink` tests for each sensor, its `SensorSource`:
# the accelerometer and the gyroscope of the Inertial Measurement Unit (IMU), the magnetometer, and the
# barometer's pressure, altitude and temperature.
IMU = 0b0000000111111
MAG = 0b0000111000000
BARO = 0b1101000000000
EVERY_FIELD = 0x1FFF
# The ticks a run steps before and after the ones a test counts, so the fake has read every message of the window.
MARGIN = 10

IMU_PRIM = sv.prim("Imu0", "NexusImuAPI")


def _receive(tmp_path, prims: str, ticks: int) -> tuple[list[int], int]:
    """Fly the fixture with `prims` against the fake PX4, and return what it received over `ticks` ticks.

    Returns:
        The `fields_updated` mask of each `HIL_SENSOR` of the window, and how many `HIL_GPS` came in it.
    """
    loop = sv.build(
        sv.vehicle(tmp_path, prims, px4=True), components=default_registry(), peers={"px4_sitl": sv.KeepingFake}
    )
    fake = loop.peers[0]
    sv.steps(loop, ticks + 2 * MARGIN)
    sensor = [msg for msg in fake.last.every if msg.get_type() == "HIL_SENSOR"]
    window = sensor[MARGIN : MARGIN + ticks]
    start, end = window[0].time_usec, sensor[MARGIN + ticks].time_usec
    gps = sum(msg.get_type() == "HIL_GPS" and start <= msg.time_usec < end for msg in fake.last.every)
    return [msg.fields_updated for msg in window], gps


def _with(masks: list[int], bits: int) -> int:
    """How many of `masks` set every bit of `bits`."""
    return sum(mask & bits == bits for mask in masks)


def test_hil_sensor_sets_a_sensors_bits_only_on_a_tick_where_that_sensor_has_a_new_sample(tmp_path):
    """Over PX4 lockstep, `HIL_SENSOR` sets a sensor's bits in `fields_updated` only on a tick where that sensor has a new sample.

    Given the fake PX4 peer and a fixture vehicle with the barometer at 50 Hz and the magnetometer at
    100 Hz on a 250 Hz tick, when the run steps 1 s, then 250 `HIL_SENSOR` arrive: 50 with the barometer's
    bits, 100 with the magnetometer's and all 250 with the IMU's.
    """
    prims = (
        IMU_PRIM
        + sv.prim("Mag0", "NexusMagAPI", "float nexus:rate = 100")
        + sv.prim("Baro0", "NexusBaroAPI", "float nexus:rate = 50")
    )

    masks, _ = _receive(tmp_path, prims, 250)

    assert (len(masks), _with(masks, BARO), _with(masks, MAG), _with(masks, IMU)) == (250, 50, 100, 250)


def test_with_no_rate_declared_every_hil_sensor_sets_every_bit(tmp_path):
    """With no rate declared, every `HIL_SENSOR` sets every bit, as today.

    Given the fake PX4 peer and a fixture vehicle that authors no `nexus:rate`, when the run steps 100
    ticks, then all 100 `HIL_SENSOR` carry `fields_updated = 0x1FFF`.
    """
    prims = (
        IMU_PRIM + sv.prim("Mag0", "NexusMagAPI") + sv.prim("Baro0", "NexusBaroAPI") + sv.prim("Gps0", "NexusGpsAPI")
    )

    masks, _ = _receive(tmp_path, prims, 100)

    assert masks == [EVERY_FIELD] * 100


def test_hil_gps_goes_out_once_per_new_gps_sample(tmp_path):
    """`HIL_GPS` goes out once per new Global Positioning System (GPS) sample.

    Given the fake PX4 peer and a fixture vehicle whose GPS declares `nexus:rate = 5`, when the run steps
    2 s, then 10 `HIL_GPS` arrive.
    """
    prims = IMU_PRIM + sv.prim("Gps0", "NexusGpsAPI", "float nexus:rate = 5")

    _, gps = _receive(tmp_path, prims, 500)

    assert gps == 10


def test_a_gps_that_declares_no_rate_sends_hil_gps_every_tick(tmp_path):
    """A GPS that declares no rate sends `HIL_GPS` every tick.

    Given the fake PX4 peer and a fixture vehicle whose GPS authors no `nexus:rate`, when the run steps
    100 ticks, then 100 `HIL_GPS` arrive.
    """
    prims = IMU_PRIM + sv.prim("Gps0", "NexusGpsAPI")

    _, gps = _receive(tmp_path, prims, 100)

    assert gps == 100
