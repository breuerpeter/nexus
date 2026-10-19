"""Shared fixtures of the PX4 Software In The Loop (SITL) peer's tests."""

import pytest


@pytest.fixture(scope="module")
def px4_vehicle(tmp_path_factory) -> str:
    """The fixture vehicle a real PX4 flies, written once for the module: PX4 declared on airframe `astro_max`, and
    the four sensors it reads at the shipped vehicle's rates, the Inertial Measurement Unit (IMU) each tick, the
    magnetometer at 100 Hz, the barometer at 50 Hz and the Global Positioning System (GPS) at 30 Hz, so PX4's
    estimator converges and it arms. A run flies it in the fixture scene.
    """
    pytest.importorskip("pxr")
    from tests.usd import sensor_vehicle as sv

    sensors = (
        sv.prim("Imu", "NexusImuAPI")
        + sv.prim("Mag", "NexusMagAPI", "float nexus:rate = 100")
        + sv.prim("Baro", "NexusBaroAPI", "float nexus:rate = 50")
        + sv.prim("Gps", "NexusGpsAPI", "float nexus:rate = 30")
    )
    return sv.vehicle(tmp_path_factory.mktemp("px4_vehicle"), sensors, px4=True)
