"""Captured sensor kernels: the in-graph sensors' device stages replay from a CUDA graph with noise
that dithers per replay and samples that match the eager ones. The tests are `gpu`, so they skip
without a CUDA device; each test scopes the device it needs with ``wp.ScopedDevice``, so the default
device is the same after it.
"""

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("warp")

import warp as wp

from tests.usd import sensor_vehicle as sv

pytestmark = pytest.mark.gpu


class _WarpView:
    def __init__(self, pos, quat_xyzw, lin, ang):
        bq = np.array([[*pos, *quat_xyzw]], dtype=np.float32)
        bqd = np.array([[*lin, *ang]], dtype=np.float32)
        self.body_q = wp.array(bq, dtype=wp.transform)
        self.body_qd = wp.array(bqd, dtype=wp.spatial_vector)


def _run():
    """The run's values each sensor here takes: a 4 ms tick and a site at 47.6, -122.3, 5 m."""
    from nexus_sim._src.core.interfaces import SensorRun
    from nexus_sim._src.scene import Site

    return SensorRun(seed=42, dt=0.004, site=Site(lat=47.6, lon=-122.3, alt=5.0, mag_ned=(0.21, 0.05, 0.43)))


def test_captured_sensor_noise_dithers_per_replay():
    """The device step counter must make sensor noise vary across graph replays. A frozen captured
    sensor stream, with a baked-at-capture step, reads to PX4's Extended Kalman Filter (EKF) as a stuck
    sensor -> Global Positioning System (GPS)/position fusion never starts -> won't arm, the real bug
    behind the captured PX4 flight.
    """
    from nexus_sim._src.core.schema import SimTime
    from nexus_sim._src.vehicle.sensors import ImuSensor

    with wp.ScopedDevice("cuda:0"):
        view = _WarpView((0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        s = sv.wired(ImuSensor(_run()))  # noise on, the default
        s.sample_wp(view, SimTime(0.0, 0))  # warmup: first=1, seeds prev
        with wp.ScopedCapture() as cap:
            s.sample_wp(view, SimTime(0.0, 0))
        wp.capture_launch(cap.graph)
        wp.synchronize()
        a = s.out.read()[0]
        wp.capture_launch(cap.graph)
        wp.synchronize()
        b = s.out.read()[0]
        # the accelerometer's and the gyro's noise dithers across replays
        assert not np.allclose([*a["accel"], *a["gyro"]], [*b["accel"], *b["gyro"]])


def test_captured_px4_sensors_match_eager():
    """The Hardware In The Loop (HIL) sensors' kernels, sample_wp with no host readback, join a CUDA graph,
    and a replay writes the samples an eager run writes, so the PX4 device region is capturable. Validated
    with noise off so the result is step-independent.
    """
    from nexus_sim._src.core.schema import SimTime
    from nexus_sim._src.vehicle.sensors import BaroSensor, GpsSensor, ImuSensor, MagSensor

    with wp.ScopedDevice("cuda:0"):
        view = _WarpView((1.0, -2.0, 3.0), (0.0, 0.0, 0.0, 1.0), (0.5, -0.1, 0.2), (0.3, -0.4, 0.5))

        def fresh():
            return [
                sv.wired(ImuSensor(_run(), acc_noise=0.0, gyro_noise=0.0)),
                sv.wired(MagSensor(_run(), noise=(0.0, 0.0, 0.0))),
                sv.wired(BaroSensor(_run(), noise=0.0)),
                sv.wired(GpsSensor(_run())),
            ]

        eager = fresh()
        for s in eager:
            s.sample_wp(view, SimTime(0.0, 0))
        eager_imu, eager_mag, eager_baro, eager_gps = (s.out.read()[0] for s in eager)

        cap_sensors = fresh()
        # warmup: seeds the Inertial Measurement Unit (IMU) finite-diff prev so capture runs with first=0
        for s in cap_sensors:
            s.sample_wp(view, SimTime(0.0, 0))
        with wp.ScopedCapture() as cap:
            for s in cap_sensors:
                s.sample_wp(view, SimTime(0.0, 0))
        wp.capture_launch(cap.graph)
        wp.synchronize()
        imu, mag, baro, gps = (s.out.read()[0] for s in cap_sensors)

        # noise off + static view -> deterministic, step-independent; captured == eager
        np.testing.assert_allclose(imu["gyro"], eager_imu["gyro"], atol=1e-6)
        np.testing.assert_allclose(mag["field"], eager_mag["field"])
        assert float(baro["pressure"]) == pytest.approx(float(eager_baro["pressure"]))
        assert [float(gps[f]) for f in ("lat", "lon", "alt")] == pytest.approx(
            [float(eager_gps[f]) for f in ("lat", "lon", "alt")]
        )
