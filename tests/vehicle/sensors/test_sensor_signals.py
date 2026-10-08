"""Each sensor's output is a signal of its own type, cameras included: a reader reads it as a signal, with
the time of its sample, and holds no sensor.

Real builds of the fixture vehicle in ``tests/usd/sensor_vehicle.py``, with the sensor prims a test adds and a
stand-in estimator that the vehicle declares on a scope of its own and whose host stage keeps what each signal
it reads holds. The Kit peer maps to its fake, and the stand-in docker daemon keeps any container a run
would start from reaching a real one. Each test scopes the device it uses, so the default device is the
same after it. Skipped without newton or pxr.
"""

import functools

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import warp as wp

from nexus_sim._src.core.schema import BaroSample, GpsSample, Image, ImuSample, MagSample, PointCloud
from nexus_sim._src.core.signals import Signal
from nexus_sim._src.peers.kit.fake import KitFake
from tests.peers.kit.conftest import daemon  # noqa: F401
from tests.usd import sensor_vehicle as sv

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]

# Where a falling run starts: 2 m over the ground, level and nose north, so its Forward Right Down (FRD)
# body axes are the North East Down (NED) axes, and it falls 8 mm in 10 ticks.
FALL_FROM = (0.0, 0.0, 2.0)
# The default site, Woodinville: its origin, and the World Magnetic Model (WMM) field there as an NED
# vector in gauss, from PX4's coarse table: declination 15.5°, inclination 69.4°, strength 0.539.
WOODINVILLE = (47.747944, -122.163917)
WOODINVILLE_NED = (0.182, 0.051, 0.504)
FIELD_TOL = 0.015
# The International Standard Atmosphere's pressure 2 m over mean sea level, hPa: 12 Pa less per metre
# than the 1013.25 hPa at sea level.
PRESSURE_AT_2_M = 1013.01

# One Inertial Measurement Unit (IMU), magnetometer, barometer and Global Positioning System (GPS)
# receiver on the base body, each with no noise, so a reading is the value the site and the body give it.
QUIET = (
    sv.prim("Imu0", "NexusImuAPI", "float nexus:accNoise = 0\nfloat nexus:gyroNoise = 0")
    + sv.prim("Mag0", "NexusMagAPI", "float3 nexus:noise = (0, 0, 0)")
    + sv.prim("Baro0", "NexusBaroAPI", "float nexus:noise = 0")
    + sv.prim("Gps0", "NexusGpsAPI")
)
# A camera, a thermal camera and a lidar on the base body, each at 25 Hz.
WIDTH, HEIGHT = 64, 48
RTX = (
    sv.prim(
        "Cam",
        "NexusCameraAPI",
        f"int nexus:width = {WIDTH}\nint nexus:height = {HEIGHT}\nfloat nexus:rate = 25",
        kind="Camera",
    )
    + sv.prim(
        "Ir",
        "NexusThermalCameraAPI",
        f"int nexus:width = {WIDTH}\nint nexus:height = {HEIGHT}\nfloat nexus:rate = 25",
        kind="Camera",
    )
    + sv.prim("Lidar", "NexusLidarAPI", "float nexus:rate = 25", kind="OmniLidar")
)
POINT = (1.0, 2.0, 3.0)


def _field(sample, name: str) -> np.ndarray:
    """The field `name` of a sensor's sample, as a flat array of floats."""
    return np.asarray(sample[name], dtype=float).reshape(-1)


def _close(value, expected, tol: float) -> bool:
    return bool(np.allclose(value, expected, rtol=0.0, atol=tol))


@pytest.mark.parametrize("device", DEVICES)
def test_an_estimator_reads_the_imu_magnetometer_barometer_and_gps_each_as_a_signal_of_its_own_type_with_its_time(
    device, tmp_path
):
    """An estimator reads the output of an IMU, a magnetometer, a barometer and a GPS, each a signal of its sensor's own type, with the time of its sample.

    Given the falling fixture vehicle, level and nose north, with one quiet Inertial Measurement Unit (IMU),
    magnetometer, barometer and Global Positioning System (GPS) receiver, and a stand-in estimator whose host
    stage reads the four sensors' signals, when the run steps 10 ticks, eagerly on the CPU device and captured
    on a CUDA device, then on each tick the estimator reads a gyro and, from the second tick, an accelerometer
    of zero within 1e-3, the site's magnetic field, the pressure at the vehicle's height and the site's latitude
    and longitude, each with that tick's sim time.
    """
    reader = sv.estimator(
        Signal("imu", ImuSample, shape=(1,)),
        Signal("mag", MagSample, shape=(1,)),
        Signal("baro", BaroSample, shape=(1,)),
        Signal("gps", GpsSample, shape=(1,)),
    )
    with wp.ScopedDevice(device):
        loop = sv.build(
            sv.vehicle(tmp_path, QUIET, estimator=""),
            device="cpu" if device == "cpu" else "cuda",
            fall_from=FALL_FROM,
            components=sv.components(StandInEstimatorAPI=reader),
        )
        sv.steps(loop, 10)
    read = [
        (
            _close(_field(imu, "gyro"), 0.0, 1e-3),
            tick == 0 or _close(_field(imu, "accel"), 0.0, 1e-3),
            _close(_field(mag, "field"), WOODINVILLE_NED, FIELD_TOL),
            _close(_field(baro, "pressure"), PRESSURE_AT_2_M, 0.005),
            _close([*_field(gps, "lat"), *_field(gps, "lon")], WOODINVILLE, 1e-4),
            _close([_field(s, "time") for s in (imu, mag, baro, gps)], t, 1e-6),
        )
        for tick, (t, imu, mag, baro, gps) in enumerate(reader.kept)
    ]

    assert read == [(True,) * 6] * 10, reader.kept


def _firsts(kept: list, column: int) -> list[tuple[float, object]]:
    """Each value a stand-in estimator's `kept` holds in `column`, once, with the sim time of the tick that first
    held it, in the order it came. A value is new when its time is: a frame or a scan stays until the next.
    """
    firsts, seen = [], set()
    for row in kept:
        value = row[column]
        if value is not None and value.time not in seen:
            seen.add(value.time)
            firsts.append((row[0], value))
    return firsts


@pytest.mark.usefixtures("warp_cpu", "daemon")
def test_each_frame_of_a_camera_and_a_thermal_camera_and_each_lidar_scan_reaches_its_reader_as_a_signal(tmp_path):
    """Each frame of a camera and of a thermal camera, and each scan of a lidar as the Kit peer sends it, reaches its reader as a signal, with the sim time the frame or scan shows.

    Given the fixture vehicle with a camera, a thermal camera and a lidar at 25 Hz each, the Kit peer mapped
    to its fake scanning the one point `(1, 2, 3)`, and a stand-in estimator whose host stage reads the three
    sensors' signals, when the run steps 1 s of sim time, then the estimator reads 23 frames of each camera,
    every one the run asked for but the two still in flight when it ends, give or take one, each of its
    camera's width and height, and 23 scans that hold `(1, 2, 3)`, give or take one, each with the sim time it
    shows, no later than the tick that hands it over.
    """
    reader = sv.estimator(Signal("camera", Image), Signal("thermal_camera", Image), Signal("lidar", PointCloud))
    loop = sv.build(
        sv.vehicle(tmp_path, RTX, estimator=""),
        peers={"kit": functools.partial(KitFake, points=[POINT])},
        components=sv.components(StandInEstimatorAPI=reader),
    )
    sv.steps(loop, 250)  # 1 s of 0.004 s ticks
    cameras = [_firsts(reader.kept, column) for column in (1, 2)]
    scans = _firsts(reader.kept, 3)
    read = [
        *(
            (
                22 <= len(frames) <= 24,
                all(frame.pixels.shape[:2] == (HEIGHT, WIDTH) for _, frame in frames),
                all(frame.time <= t for t, frame in frames),
            )
            for frames in cameras
        ),
        (
            22 <= len(scans) <= 24,
            all(_close(scan.points, [POINT], 1e-6) for _, scan in scans),
            all(scan.time <= t for t, scan in scans),
        ),
    ]

    assert read == [(True, True, True)] * 3, [len(frames) for frames in (*cameras, scans)]
