"""Warp-native sensors: math parity compared to the canonical ``transform`` oracle with noise
off, and run-to-run determinism on the Warp random number generator, the per-sensor
noise field.
"""

import math

import numpy as np
import pytest

pytest.importorskip("warp")
import warp as wp

# the canonical host reference / oracle
from nexus_sim._src import transform
from nexus_sim._src.core.interfaces import SensorRun
from nexus_sim._src.core.schema import Measurement, SimTime
from nexus_sim._src.core.seedtree import SeedTree
from nexus_sim._src.scene import Site
from nexus_sim._src.vehicle.sensors import BaroSensor, GpsSensor, ImuSensor, MagSensor
from tests.usd import sensor_vehicle as sv

pytestmark = pytest.mark.usefixtures("warp_cpu")


class _WarpView:
    """Minimal state stand-in exposing body_q/body_qd for one body."""

    def __init__(self, pos, quat_xyzw, lin, ang):
        bq = np.array([[*pos, *quat_xyzw]], dtype=np.float32)
        bqd = np.array([[*lin, *ang]], dtype=np.float32)
        self.body_q = wp.array(bq, dtype=wp.transform)
        self.body_qd = wp.array(bqd, dtype=wp.spatial_vector)


# a non-trivial, normalized attitude so the world<->body rotation is actually exercised
_Q = list(np.array([0.1, -0.2, 0.3, 0.9], dtype=np.float64) / np.linalg.norm([0.1, -0.2, 0.3, 0.9]))
# The site values every sensor here is built with: gravity along world -Z, a field in NED gauss.
_GRAVITY_WORLD = (0.0, 0.0, -9.81)
_MAG_NED = (0.21, 0.05, 0.43)


def _run(mag_ned=_MAG_NED, body: int = 0) -> SensorRun:
    """The run's values each sensor here takes: seed 42, a 4 ms tick and a site at 47.6, -122.3, 5 m."""
    site = Site(lat=47.6, lon=-122.3, alt=5.0, mag_ned=mag_ned)
    return SensorRun(seed=SeedTree(42).seed_for("sensor"), dt=0.004, site=site, body=body)


def test_imu_gyro_parity():
    view = _WarpView((0.0, 0.0, 1.0), _Q, (0.0, 0.0, 0.0), (0.3, -0.4, 0.5))
    s = sv.wired(ImuSensor(_run(), acc_noise=0.0, gyro_noise=0.0))
    meas = Measurement()
    s.sample(view, SimTime(0.0, 0), meas)
    wp.synchronize()
    q = transform.quat_from_xyzw(_Q)
    gyro_ref = transform.world_to_body(q, (0.3, -0.4, 0.5))
    np.testing.assert_allclose([meas.xgyro, meas.ygyro, meas.zgyro], gyro_ref, atol=1e-5)


def test_imu_accel_finite_diff_parity():
    s = sv.wired(ImuSensor(_run(), acc_noise=0.0, gyro_noise=0.0))
    meas = Measurement()
    v1, v2 = (0.0, 0.0, 0.0), (0.04, -0.02, 0.06)  # velocity change over one tick
    s.sample(_WarpView((0, 0, 1), _Q, v1, (0.3, -0.4, 0.5)), SimTime(0.0, 0), meas)  # first: accel 0
    s.sample(_WarpView((0, 0, 1), _Q, v2, (0.3, -0.4, 0.5)), SimTime(0.004, 1), meas)
    wp.synchronize()
    q = transform.quat_from_xyzw(_Q)
    acc_com = (np.array(v2) - np.array(v1)) / 0.004
    grav_b = transform.world_to_body(q, _GRAVITY_WORLD)
    acc_b = transform.world_to_body(q, acc_com)  # zero mount -> no lever arm
    ref = [acc_b[i] - grav_b[i] for i in range(3)]
    np.testing.assert_allclose([meas.xacc, meas.yacc, meas.zacc], ref, atol=1e-4)


def test_mag_parity():
    view = _WarpView((0, 0, 1), _Q, (0, 0, 0), (0, 0, 0))
    s = sv.wired(MagSensor(_run(), offset=(0.01, -0.02, 0.0), noise=(0.0, 0.0, 0.0)))
    meas = Measurement()
    s.sample(view, SimTime(0.0, 0), meas)
    wp.synchronize()
    q = transform.quat_from_xyzw(_Q)
    mag_b = transform.world_to_body(q, transform.mag_ned_to_world(_MAG_NED))
    ref = [mag_b[0] + 0.01, mag_b[1] - 0.02, mag_b[2] + 0.0]
    np.testing.assert_allclose([meas.xmag, meas.ymag, meas.zmag], ref, atol=1e-5)


def test_baro_and_gps_parity():
    pos = (12.0, -7.0, 30.0)
    view = _WarpView(pos, _Q, (1.5, -0.5, 0.2), (0, 0, 0))
    sv.wired(BaroSensor(_run(), noise=0.0)).sample(view, SimTime(0.0, 0), m := Measurement())
    sv.wired(GpsSensor(_run())).sample(view, SimTime(0.0, 0), m)
    wp.synchronize()
    assert m.abs_pressure == pytest.approx(1013.25 * (1 - 2.25577e-5 * 30.0) ** 5.25588, abs=1e-3)
    assert m.pressure_alt == pytest.approx(30.0, abs=1e-4)
    lat, lon, alt = transform.gps_from_local(47.6, -122.3, 5.0, pos)
    assert (m.lat_deg, m.lon_deg, m.alt_m) == pytest.approx((lat, lon, alt), abs=1e-6)
    assert (m.vn, m.ve, m.vd) == pytest.approx((1.5, 0.5, -0.2))  # world->North East Down (NED): ve=-vy, vd=-vz
    assert m.ground_speed == pytest.approx(math.hypot(1.5, -0.5), abs=1e-5)


def test_world_to_ned_is_a_proper_rotation():
    """The NED basis images recovered *from* the sensors must obey n x e = d.

    Read out of the Global Positioning System (GPS) position map, cross-checked against the GPS
    velocity map and the magnetometer, so a sign flip in any single one of the three breaks this,
    see GH #61.
    """
    ref_lat, ref_lon, ref_alt = 47.6, -122.3, 5.0
    inv_lon_scale = 1.0 / (111000.0 * math.cos(math.radians(ref_lat)))

    def gps_at(pos, vel=(0.0, 0.0, 0.0)):
        sv.wired(GpsSensor(_run())).sample(
            _WarpView(pos, (0.0, 0.0, 0.0, 1.0), vel, (0, 0, 0)), SimTime(0.0, 0), m := Measurement()
        )
        wp.synchronize()
        return m

    # The world offset that moves one metre along each NED axis, read out of the position map.
    assert gps_at((1.0, 0.0, 0.0)).lat_deg == pytest.approx(ref_lat + 1.0 / 111000.0, abs=1e-12)  # n = +x
    assert gps_at((0.0, -1.0, 0.0)).lon_deg == pytest.approx(ref_lon + inv_lon_scale, abs=1e-12)  # e = -y
    assert gps_at((0.0, 0.0, -1.0)).alt_m == pytest.approx(ref_alt - 1.0, abs=1e-9)  # d = -z
    n_world, e_world, d_world = (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)

    # The velocity map's east axis must be the position map's, or the Extended Kalman Filter (EKF) gets
    # GPS position and GPS velocity disagreeing in sign: the #61 runaway.
    assert gps_at((0.0, 0.0, 0.0), vel=e_world).ve == pytest.approx(1.0, abs=1e-9)

    # The mag map's basis images, read out of the magnetometer at identity attitude.
    for ned, want in ((n_world, n_world), ((0.0, 1.0, 0.0), e_world), ((0.0, 0.0, 1.0), d_world)):
        sv.wired(MagSensor(_run(mag_ned=ned), noise=(0.0, 0.0, 0.0))).sample(
            _WarpView((0, 0, 1), (0.0, 0.0, 0.0, 1.0), (0, 0, 0), (0, 0, 0)), SimTime(0.0, 0), m := Measurement()
        )
        wp.synchronize()
        np.testing.assert_allclose([m.xmag, m.ymag, m.zmag], want, atol=1e-6)

    np.testing.assert_allclose(np.cross(n_world, e_world), d_world, atol=1e-12)


def test_a_magnetometer_barometer_and_gps_read_the_body_the_run_names():
    """A magnetometer, a barometer and a GPS receiver read the body the run's values name, not body 0.

    Given two bodies, the first at the origin and level, the second 30 m up and yawed 90 degrees, and each
    sensor built for body 1, when sampled, then the barometer and the receiver report 30 m, and the
    magnetometer the field in the second body's axes.
    """
    s2 = math.sqrt(2.0) / 2.0
    view = _WarpView((0, 0, 0), (0.0, 0.0, 0.0, 1.0), (0, 0, 0), (0, 0, 0))
    view.body_q = wp.array(
        np.array([[0, 0, 0, 0, 0, 0, 1], [0, 0, 30, 0, 0, s2, s2]], dtype=np.float32), dtype=wp.transform
    )
    view.body_qd = wp.array(np.zeros((2, 6), dtype=np.float32), dtype=wp.spatial_vector)
    m = Measurement()
    for sensor in (
        sv.wired(MagSensor(_run(body=1), noise=(0.0, 0.0, 0.0))),
        sv.wired(BaroSensor(_run(body=1), noise=0.0)),
        sv.wired(GpsSensor(_run(body=1))),
    ):
        sensor.sample(view, SimTime(0.0, 0), m)
    wp.synchronize()

    # The field in body 1's axes, as in the known-attitude case that follows: (-0.05, -0.21, -0.43).
    np.testing.assert_allclose(
        [m.pressure_alt, m.alt_m, m.xmag, m.ymag, m.zmag], [30.0, 35.0, -0.05, -0.21, -0.43], atol=1e-5
    )


def test_a_barometer_whose_prim_sits_off_its_bodys_origin_fails_and_names_the_prim():
    """A sensor that models no mount offset fails when its prim sits off its body's origin, and names the prim."""
    run = SensorRun(seed=1, dt=0.004, site=_run().site, mount=(0.1, 0.0, 0.0), path="/Vehicle/body/Baro")

    with pytest.raises(ValueError, match="/Vehicle/body/Baro"):
        BaroSensor(run)


def test_mag_known_attitude():
    """One hand-computed case, no oracle: 90° about world +z, so the map can't drift with it."""
    s2 = math.sqrt(2.0) / 2.0
    view = _WarpView((0, 0, 1), (0.0, 0.0, s2, s2), (0, 0, 0), (0, 0, 0))  # yaw 90° about +z
    sv.wired(MagSensor(_run(), noise=(0.0, 0.0, 0.0))).sample(view, SimTime(0.0, 0), m := Measurement())
    wp.synchronize()
    # mag_ned (0.21, 0.05, 0.43) -> world (n, -e, -d) = (0.21, -0.05, -0.43); R(q)^-1 maps
    # (x, y, z) -> (y, -x, z), giving body (-0.05, -0.21, -0.43).
    np.testing.assert_allclose([m.xmag, m.ymag, m.zmag], [-0.05, -0.21, -0.43], atol=1e-6)


def test_seed_for_is_deterministic_and_named():
    a, b = SeedTree(42), SeedTree(42)
    assert a.seed_for("imu") == b.seed_for("imu")  # reproducible across runs
    assert a.seed_for("imu") != a.seed_for("mag")  # independent per consumer
    assert SeedTree(7).seed_for("imu") != SeedTree(8).seed_for("imu")  # depends on root seed


def test_imu_noise_is_run_to_run_bit_identical():
    view = _WarpView((0, 0, 1), _Q, (0.04, 0.0, 0.0), (0.3, -0.4, 0.5))

    def run():
        s = sv.wired(ImuSensor(_run()))  # default noise on
        m = Measurement()
        s.sample(view, SimTime(0.0, 0), m)
        s.sample(view, SimTime(0.004, 1), m)
        return (m.xacc, m.yacc, m.zacc, m.xgyro, m.ygyro, m.zgyro)

    assert run() == run()  # same seed + step -> bit-for-bit the same Warp noise


@pytest.mark.parametrize("cls", [ImuSensor, MagSensor, BaroSensor, GpsSensor])
def test_each_analytic_sensors_sample_carries_the_sim_time_the_signal_time_holds(cls):
    """Each analytic sensor's sample carries the sim time the signal `time` holds when it samples.

    Given an Inertial Measurement Unit (IMU), a magnetometer, a barometer or a Global Positioning System (GPS)
    receiver, wired alone with the tick's sim time at 1.25 s, when it samples a body at rest, then its sample's
    time is 1.25 s.
    """
    view = _WarpView((0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    sensor = sv.wired(cls(_run()), time=1.25)
    sensor.sample_wp(view, SimTime(1.25, 0))

    assert float(sensor.out.read()[0]["time"]) == 1.25
