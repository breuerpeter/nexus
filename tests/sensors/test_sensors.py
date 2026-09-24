"""Warp-native sensors, Stage 3: math parity compared to the canonical ``transform`` oracle with noise
off, and run-to-run determinism on the Warp random number generator, the re-baselined NFR-1
noise field.
"""

import math

import numpy as np
import pytest

pytest.importorskip("warp")
import warp as wp

# the canonical host reference / oracle
from nexus._src import transform
from nexus._src.core.schema import EnvSample, Measurement, SimTime
from nexus._src.core.seedtree import SeedTree
from nexus._src.vehicle.sensors import BaroSensor, GpsSensor, ImuSensor, MagSensor

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
_ENV = EnvSample(
    gravity_world=(0.0, 0.0, -9.81), mag_ned=(0.21, 0.05, 0.43), air_pressure_msl=1013.25, temperature=25.0
)


def test_imu_gyro_and_quat_parity():
    view = _WarpView((0.0, 0.0, 1.0), _Q, (0.0, 0.0, 0.0), (0.3, -0.4, 0.5))
    s = ImuSensor(SeedTree(42), dt=0.004, acc_noise=0.0, gyro_noise=0.0)
    meas = Measurement()
    s.sample(view, _ENV, SimTime(0.0, 0), meas)
    wp.synchronize()
    q = transform.quat_from_xyzw(_Q)
    gyro_ref = transform.world_to_body(q, (0.3, -0.4, 0.5))
    np.testing.assert_allclose([meas.xgyro, meas.ygyro, meas.zgyro], gyro_ref, atol=1e-5)
    np.testing.assert_allclose(meas.quat_wxyz, transform.quat_wxyz(q), atol=1e-6)
    assert (meas.rollspeed, meas.pitchspeed, meas.yawspeed) == (meas.xgyro, meas.ygyro, meas.zgyro)


def test_imu_accel_finite_diff_parity():
    s = ImuSensor(SeedTree(42), dt=0.004, acc_noise=0.0, gyro_noise=0.0)
    meas = Measurement()
    v1, v2 = (0.0, 0.0, 0.0), (0.04, -0.02, 0.06)  # velocity change over one tick
    s.sample(_WarpView((0, 0, 1), _Q, v1, (0.3, -0.4, 0.5)), _ENV, SimTime(0.0, 0), meas)  # first: accel 0
    s.sample(_WarpView((0, 0, 1), _Q, v2, (0.3, -0.4, 0.5)), _ENV, SimTime(0.004, 1), meas)
    wp.synchronize()
    q = transform.quat_from_xyzw(_Q)
    acc_com = (np.array(v2) - np.array(v1)) / 0.004
    grav_b = transform.world_to_body(q, _ENV.gravity_world)
    acc_b = transform.world_to_body(q, acc_com)  # zero mount -> no lever arm
    ref = [acc_b[i] - grav_b[i] for i in range(3)]
    np.testing.assert_allclose([meas.xacc, meas.yacc, meas.zacc], ref, atol=1e-4)


def test_mag_parity():
    view = _WarpView((0, 0, 1), _Q, (0, 0, 0), (0, 0, 0))
    s = MagSensor(SeedTree(42), mag_offset=(0.01, -0.02, 0.0), noise=(0.0, 0.0, 0.0))
    meas = Measurement()
    s.sample(view, _ENV, SimTime(0.0, 0), meas)
    wp.synchronize()
    q = transform.quat_from_xyzw(_Q)
    mag_b = transform.world_to_body(q, transform.mag_ned_to_world(_ENV.mag_ned))
    ref = [mag_b[0] + 0.01, mag_b[1] - 0.02, mag_b[2] + 0.0]
    np.testing.assert_allclose([meas.xmag, meas.ymag, meas.zmag], ref, atol=1e-5)


def test_baro_and_gps_parity():
    pos = (12.0, -7.0, 30.0)
    view = _WarpView(pos, _Q, (1.5, -0.5, 0.2), (0, 0, 0))
    BaroSensor(SeedTree(42), noise=0.0).sample(view, _ENV, SimTime(0.0, 0), m := Measurement())
    GpsSensor(47.6, -122.3, 5.0).sample(view, _ENV, SimTime(0.0, 0), m)
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
        GpsSensor(ref_lat, ref_lon, ref_alt).sample(
            _WarpView(pos, (0.0, 0.0, 0.0, 1.0), vel, (0, 0, 0)), _ENV, SimTime(0.0, 0), m := Measurement()
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
        MagSensor(SeedTree(42), mag_offset=(0.0, 0.0, 0.0), noise=(0.0, 0.0, 0.0)).sample(
            _WarpView((0, 0, 1), (0.0, 0.0, 0.0, 1.0), (0, 0, 0), (0, 0, 0)),
            EnvSample(
                gravity_world=_ENV.gravity_world,
                mag_ned=ned,
                air_pressure_msl=_ENV.air_pressure_msl,
                temperature=_ENV.temperature,
            ),
            SimTime(0.0, 0),
            m := Measurement(),
        )
        wp.synchronize()
        np.testing.assert_allclose([m.xmag, m.ymag, m.zmag], want, atol=1e-6)

    np.testing.assert_allclose(np.cross(n_world, e_world), d_world, atol=1e-12)


def test_mag_known_attitude():
    """One hand-computed case, no oracle: 90° about world +z, so the map can't drift with it."""
    s2 = math.sqrt(2.0) / 2.0
    view = _WarpView((0, 0, 1), (0.0, 0.0, s2, s2), (0, 0, 0), (0, 0, 0))  # yaw 90° about +z
    MagSensor(SeedTree(42), mag_offset=(0.0, 0.0, 0.0), noise=(0.0, 0.0, 0.0)).sample(
        view, _ENV, SimTime(0.0, 0), m := Measurement()
    )
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
        s = ImuSensor(SeedTree(42), dt=0.004)  # default noise on
        m = Measurement()
        s.sample(view, _ENV, SimTime(0.0, 0), m)
        s.sample(view, _ENV, SimTime(0.004, 1), m)
        return (m.xacc, m.yacc, m.zacc, m.xgyro, m.ygyro, m.zgyro)

    assert run() == run()  # same seed + step -> bit-for-bit the same Warp noise
