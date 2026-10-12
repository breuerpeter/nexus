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
from nexus_sim._src.core.schema import SimTime
from nexus_sim._src.core.seedtree import SeedTree
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
# The world's values every sensor here takes: gravity along world -Z, a field in North East Down (NED) gauss.
_GRAVITY_WORLD = (0.0, 0.0, -9.81)
_MAG_NED = sv.FIELD_NED
_SEED = SeedTree(42).seed_for("sensor")


def _on(body: int = 0, **world) -> tuple:
    """The model each sensor here takes, with the world's values `world` set by hand, and the label of
    the site on body `body`: ``cls(*_on(), ...)``.
    """
    model, sites = sv.model(bodies=body + 1, **world)
    return model, sites[body]


def _sample(sensor, view, t: SimTime | None = None):
    """What `sensor`, wired, writes to its output signal when it samples the state `view` at sim time `t`, the
    start by default: the signal `time` holds `t` when it samples.
    """
    if t is not None:
        sensor.time.write([t.sim_time])
    sensor.sample_wp(view, t or SimTime(0.0, 0))
    return sensor.out.read()[0]


def test_imu_gyro_parity():
    view = _WarpView((0.0, 0.0, 1.0), _Q, (0.0, 0.0, 0.0), (0.3, -0.4, 0.5))
    sample = _sample(sv.wired(ImuSensor(*_on(), seed=_SEED, acc_noise=0.0, gyro_noise=0.0)), view)
    q = transform.quat_from_xyzw(_Q)
    gyro_ref = transform.world_to_body(q, (0.3, -0.4, 0.5))
    np.testing.assert_allclose(sample["gyro"], gyro_ref, atol=1e-5)


def test_imu_accel_finite_diff_parity():
    s = sv.wired(ImuSensor(*_on(), seed=_SEED, acc_noise=0.0, gyro_noise=0.0))
    v1, v2 = (0.0, 0.0, 0.0), (0.04, -0.02, 0.06)  # velocity change over one tick
    _sample(s, _WarpView((0, 0, 1), _Q, v1, (0.3, -0.4, 0.5)))  # first: accel 0
    sample = _sample(s, _WarpView((0, 0, 1), _Q, v2, (0.3, -0.4, 0.5)), SimTime(0.004, 1))
    q = transform.quat_from_xyzw(_Q)
    acc_com = (np.array(v2) - np.array(v1)) / 0.004
    grav_b = transform.world_to_body(q, _GRAVITY_WORLD)
    acc_b = transform.world_to_body(q, acc_com)  # zero mount -> no lever arm
    ref = [acc_b[i] - grav_b[i] for i in range(3)]
    np.testing.assert_allclose(sample["accel"], ref, atol=1e-4)


def test_mag_parity():
    view = _WarpView((0, 0, 1), _Q, (0, 0, 0), (0, 0, 0))
    sample = _sample(sv.wired(MagSensor(*_on(), seed=_SEED, offset=(0.01, -0.02, 0.0), noise=(0.0, 0.0, 0.0))), view)
    q = transform.quat_from_xyzw(_Q)
    mag_b = transform.world_to_body(q, transform.mag_ned_to_world(_MAG_NED))
    ref = [mag_b[0] + 0.01, mag_b[1] - 0.02, mag_b[2] + 0.0]
    np.testing.assert_allclose(sample["field"], ref, atol=1e-5)


def test_baro_and_gps_parity():
    pos = (12.0, -7.0, 30.0)
    view = _WarpView(pos, _Q, (1.5, -0.5, 0.2), (0, 0, 0))
    baro = _sample(sv.wired(BaroSensor(*_on(), seed=_SEED, noise=0.0)), view)
    gps = _sample(sv.wired(GpsSensor(*_on(), seed=_SEED)), view)
    assert float(baro["pressure"]) == pytest.approx(1013.25 * (1 - 2.25577e-5 * 30.0) ** 5.25588, abs=1e-3)
    assert float(baro["altitude"]) == pytest.approx(30.0, abs=1e-4)
    lat, lon, alt = transform.gps_from_local(47.6, -122.3, 5.0, pos)
    assert (float(gps["lat"]), float(gps["lon"]), float(gps["alt"])) == pytest.approx((lat, lon, alt), abs=1e-6)
    # world -> North East Down (NED): ve = -vy, vd = -vz
    assert [float(v) for v in gps["velocity"]] == pytest.approx((1.5, 0.5, -0.2))
    assert float(gps["ground_speed"]) == pytest.approx(math.hypot(1.5, -0.5), abs=1e-5)


def test_world_to_ned_is_a_proper_rotation():
    """The NED basis images recovered *from* the sensors must obey n x e = d.

    Read out of the Global Positioning System (GPS) position map, cross-checked against the GPS
    velocity map and the magnetometer, so a sign flip in any single one of the three breaks this,
    see GH #61.
    """
    ref_lat, ref_lon, ref_alt = 47.6, -122.3, 5.0
    inv_lon_scale = 1.0 / (111000.0 * math.cos(math.radians(ref_lat)))

    def gps_at(pos, vel=(0.0, 0.0, 0.0)):
        return _sample(sv.wired(GpsSensor(*_on(), seed=_SEED)), _WarpView(pos, (0.0, 0.0, 0.0, 1.0), vel, (0, 0, 0)))

    # The world offset that moves one metre along each NED axis, read out of the position map.
    assert float(gps_at((1.0, 0.0, 0.0))["lat"]) == pytest.approx(ref_lat + 1.0 / 111000.0, abs=1e-12)  # n = +x
    assert float(gps_at((0.0, -1.0, 0.0))["lon"]) == pytest.approx(ref_lon + inv_lon_scale, abs=1e-12)  # e = -y
    assert float(gps_at((0.0, 0.0, -1.0))["alt"]) == pytest.approx(ref_alt - 1.0, abs=1e-9)  # d = -z
    n_world, e_world, d_world = (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)

    # The velocity map's east axis must be the position map's, or the Extended Kalman Filter (EKF) gets
    # GPS position and GPS velocity disagreeing in sign: the #61 runaway.
    assert float(gps_at((0.0, 0.0, 0.0), vel=e_world)["velocity"][1]) == pytest.approx(1.0, abs=1e-9)

    # The mag map's basis images, read out of the magnetometer at identity attitude.
    for ned, want in ((n_world, n_world), ((0.0, 1.0, 0.0), e_world), ((0.0, 0.0, 1.0), d_world)):
        mag = sv.wired(MagSensor(*_on(field_ned=ned), seed=_SEED, noise=(0.0, 0.0, 0.0)))
        sample = _sample(mag, _WarpView((0, 0, 1), (0.0, 0.0, 0.0, 1.0), (0, 0, 0), (0, 0, 0)))
        np.testing.assert_allclose(sample["field"], want, atol=1e-6)

    np.testing.assert_allclose(np.cross(n_world, e_world), d_world, atol=1e-12)


def test_a_magnetometer_barometer_and_gps_read_the_body_their_site_sits_on():
    """A magnetometer, a barometer and a GPS receiver read the body their site sits on, not body 0.

    Given two bodies, the first at the origin and level, the second 30 m up and yawed 90 degrees, and each
    sensor built on the second body's site, when sampled, then the barometer and the receiver report 30 m,
    and the magnetometer the field in the second body's axes.
    """
    s2 = math.sqrt(2.0) / 2.0
    view = _WarpView((0, 0, 0), (0.0, 0.0, 0.0, 1.0), (0, 0, 0), (0, 0, 0))
    view.body_q = wp.array(
        np.array([[0, 0, 0, 0, 0, 0, 1], [0, 0, 30, 0, 0, s2, s2]], dtype=np.float32), dtype=wp.transform
    )
    view.body_qd = wp.array(np.zeros((2, 6), dtype=np.float32), dtype=wp.spatial_vector)
    mag, baro, gps = (
        _sample(sensor, view)
        for sensor in (
            sv.wired(MagSensor(*_on(body=1), seed=_SEED, noise=(0.0, 0.0, 0.0))),
            sv.wired(BaroSensor(*_on(body=1), seed=_SEED, noise=0.0)),
            sv.wired(GpsSensor(*_on(body=1), seed=_SEED)),
        )
    )

    # The field in body 1's axes, as in the known-attitude case that follows: (-0.05, -0.21, -0.43).
    np.testing.assert_allclose(
        [float(baro["altitude"]), float(gps["alt"]), *mag["field"]], [30.0, 35.0, -0.05, -0.21, -0.43], atol=1e-5
    )


def test_a_barometer_whose_site_sits_off_its_bodys_origin_fails_and_names_the_site():
    """A sensor that models no mount offset fails when its site sits off its body's origin, and names the site."""
    model, site = _on(mount=(0.1, 0.0, 0.0))

    with pytest.raises(ValueError, match=site):
        BaroSensor(model, site)


@pytest.mark.parametrize("cls", [ImuSensor, MagSensor, BaroSensor, GpsSensor])
def test_an_analytic_sensor_whose_rate_is_negative_fails_and_names_the_site(cls):
    """An analytic sensor whose declared rate is negative fails, and names the site."""
    model, site = _on()

    with pytest.raises(ValueError, match=site):
        cls(model, site, rate=-1.0)


@pytest.mark.parametrize("cls", [ImuSensor, MagSensor, BaroSensor, GpsSensor])
def test_an_analytic_sensor_on_a_label_that_is_no_site_of_the_model_fails_and_names_it(cls):
    """An analytic sensor built on a label no site of the model carries fails, and names the label.

    Given a one-body model with one site, when a sensor takes the label of a prim the model never loaded,
    then it fails and the message names that label.
    """
    model, _ = _on()

    with pytest.raises(ValueError, match="/vehicle/body/Nowhere"):
        cls(model, "/vehicle/body/Nowhere")


def test_mag_known_attitude():
    """One hand-computed case, no oracle: 90° about world +z, so the map can't drift with it."""
    s2 = math.sqrt(2.0) / 2.0
    view = _WarpView((0, 0, 1), (0.0, 0.0, s2, s2), (0, 0, 0), (0, 0, 0))  # yaw 90° about +z
    sample = _sample(sv.wired(MagSensor(*_on(), seed=_SEED, noise=(0.0, 0.0, 0.0))), view)
    # mag_ned (0.21, 0.05, 0.43) -> world (n, -e, -d) = (0.21, -0.05, -0.43); R(q)^-1 maps
    # (x, y, z) -> (y, -x, z), giving body (-0.05, -0.21, -0.43).
    np.testing.assert_allclose(sample["field"], [-0.05, -0.21, -0.43], atol=1e-6)


def test_seed_for_is_deterministic_and_named():
    a, b = SeedTree(42), SeedTree(42)
    assert a.seed_for("imu") == b.seed_for("imu")  # reproducible across runs
    assert a.seed_for("imu") != a.seed_for("mag")  # independent per consumer
    assert SeedTree(7).seed_for("imu") != SeedTree(8).seed_for("imu")  # depends on root seed


def test_imu_noise_is_run_to_run_bit_identical():
    view = _WarpView((0, 0, 1), _Q, (0.04, 0.0, 0.0), (0.3, -0.4, 0.5))

    def run():
        s = sv.wired(ImuSensor(*_on(), seed=_SEED))  # default noise on
        _sample(s, view)
        sample = _sample(s, view, SimTime(0.004, 1))
        return (*sample["accel"].tolist(), *sample["gyro"].tolist())

    assert run() == run()  # same seed + step -> bit-for-bit the same Warp noise


@pytest.mark.parametrize("cls", [ImuSensor, MagSensor, BaroSensor, GpsSensor])
def test_each_analytic_sensors_sample_carries_the_sim_time_the_signal_time_holds(cls):
    """Each analytic sensor's sample carries the sim time the signal `time` holds when it samples.

    Given an Inertial Measurement Unit (IMU), a magnetometer, a barometer or a Global Positioning System (GPS)
    receiver, wired alone with the tick's sim time at 1.25 s, when it samples a body at rest, then its sample's
    time is 1.25 s.
    """
    view = _WarpView((0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    sensor = sv.wired(cls(*_on(), seed=_SEED), time=1.25)
    sensor.sample_wp(view, SimTime(1.25, 0))

    assert float(sensor.out.read()[0]["time"]) == 1.25


def _sample_times(sensor, view, ticks: int) -> list[float]:
    """The time of the sample `sensor`, wired, holds on each of `ticks` ticks of 4 ms from 0 s, as it samples `view`
    on each. The tick's time is the tick count times 4 ms, in whole nanoseconds, as the device clock keeps it.
    """
    held = []
    for tick in range(ticks):
        now = tick * 4_000_000 / 1e9
        sensor.time_ns.write([tick * 4_000_000])
        sensor.time.write([now])
        sensor.sample_wp(view, SimTime(now, tick))
        held.append(float(sensor.out.read()[0]["time"]))
    return held


@pytest.mark.parametrize("cls", [ImuSensor, MagSensor, BaroSensor, GpsSensor])
def test_each_analytic_sensor_at_a_declared_rate_samples_only_on_its_due_ticks(cls):
    """Each analytic sensor at a declared rate takes a sample only on a due tick, and its signal holds it until the next.

    Given an IMU, a magnetometer, a barometer or a GPS receiver at 50 Hz, wired alone, when it samples a body at
    rest on ten 4 ms ticks from 0 s, then its signal holds the sample of 0 s on the first five ticks and the sample
    of 0.02 s on the other five.
    """
    view = _WarpView((0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    held = _sample_times(sv.wired(cls(*_on(), seed=_SEED, rate=50.0)), view, 10)

    assert held == pytest.approx([0.0] * 5 + [0.02] * 5, abs=1e-12)


def test_a_sensor_whose_rate_divides_the_tick_rate_samples_every_nth_tick():
    """A sensor whose rate divides the tick rate samples every n-th tick.

    Given a barometer at 50 Hz, wired alone, when it samples a body at rest on 2000 ticks of 4 ms, 8 s, then it
    takes a new sample on every fifth tick and on no other.
    """
    view = _WarpView((0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    held = _sample_times(sv.wired(BaroSensor(*_on(), seed=_SEED, rate=50.0)), view, 2000)

    new = [tick for tick in range(len(held)) if tick == 0 or held[tick] != held[tick - 1]]
    assert new == list(range(0, 2000, 5))


def test_an_imu_slower_than_the_tick_differences_the_velocity_over_the_time_since_its_last_sample():
    """An IMU slower than the tick finite-differences the velocity over the time since its last sample.

    Given an IMU at 125 Hz with no noise, wired alone, on a level body that falls from rest, when it samples on
    three 4 ms ticks, then its sample on the third reads zero specific force: the velocity change of two ticks
    over the time of two ticks.
    """
    sensor = sv.wired(ImuSensor(*_on(), seed=_SEED, acc_noise=0.0, gyro_noise=0.0, rate=125.0))
    for tick in range(3):
        fall = (0.0, 0.0, _GRAVITY_WORLD[2] * 0.004 * tick)  # the velocity after `tick` ticks of free fall
        sensor.time_ns.write([tick * 4_000_000])
        sensor.time.write([0.004 * tick])
        sample = _sample(
            sensor, _WarpView((0, 0, 1), (0.0, 0.0, 0.0, 1.0), fall, (0, 0, 0)), SimTime(0.004 * tick, tick)
        )

    np.testing.assert_allclose(sample["accel"], [0.0, 0.0, 0.0], atol=1e-4)
