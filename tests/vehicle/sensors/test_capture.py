"""Captured sensor kernels: the in-graph sensors' device stages replay from a CUDA graph with noise
that dithers per replay and samples that match the eager ones. Auto-skips without a CUDA device;
each test scopes the device it needs with ``wp.ScopedDevice``, so the default device is the same
after it.
"""

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("warp")

import warp as wp


class _WarpView:
    def __init__(self, pos, quat_xyzw, lin, ang):
        bq = np.array([[*pos, *quat_xyzw]], dtype=np.float32)
        bqd = np.array([[*lin, *ang]], dtype=np.float32)
        self.body_q = wp.array(bq, dtype=wp.transform)
        self.body_qd = wp.array(bqd, dtype=wp.spatial_vector)


def test_captured_sensor_noise_dithers_per_replay():
    """The device step counter must make sensor noise vary across graph replays. A frozen captured
    sensor stream, with a baked-at-capture step, reads to PX4's Extended Kalman Filter (EKF) as a stuck
    sensor -> Global Positioning System (GPS)/position fusion never starts -> won't arm, the real bug
    behind the captured PX4 flight. Auto-skips without CUDA.
    """
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    from nexus._src.core.schema import SimTime
    from nexus._src.core.seedtree import SeedTree
    from nexus._src.vehicle.sensors import ImuSensor

    with wp.ScopedDevice("cuda:0"):
        view = _WarpView((0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        s = ImuSensor(SeedTree(42), 0.004)  # noise on, the default
        s.sample_wp(view, SimTime(0.0, 0))  # warmup: first=1, seeds prev
        with wp.ScopedCapture() as cap:
            s.sample_wp(view, SimTime(0.0, 0))
        wp.capture_launch(cap.graph)
        wp.synchronize()
        a = s._out.numpy().copy()
        wp.capture_launch(cap.graph)
        wp.synchronize()
        b = s._out.numpy().copy()
        assert not np.allclose(a[:6], b[:6])  # acc+gyro noise dithers across replays
        np.testing.assert_allclose(a[6:], b[6:], atol=1e-6)  # quat stable, static state


def test_captured_px4_sensors_match_eager():
    """The Warp Measurement buffer, B: the Hardware In The Loop (HIL) sensors' kernels, sample_wp with no
    host readback, join a CUDA graph; replay + one read() reproduces the eager sample() values, so the
    PX4 device region is capturable. Validated with noise off so the result is step-independent.
    """
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    from nexus._src.core.schema import Measurement, SimTime
    from nexus._src.core.seedtree import SeedTree
    from nexus._src.vehicle.sensors import BaroSensor, GpsSensor, ImuSensor, MagSensor

    with wp.ScopedDevice("cuda:0"):
        view = _WarpView((1.0, -2.0, 3.0), (0.0, 0.0, 0.0, 1.0), (0.5, -0.1, 0.2), (0.3, -0.4, 0.5))

        def fresh():
            return [
                ImuSensor(SeedTree(42), 0.004, acc_noise=0.0, gyro_noise=0.0),
                MagSensor(SeedTree(42), (0.21, 0.05, 0.43), noise=(0.0, 0.0, 0.0)),
                BaroSensor(SeedTree(42), noise=0.0),
                GpsSensor(47.6, -122.3, 5.0),
            ]

        eager = fresh()
        m_eager = Measurement()
        for s in eager:
            s.sample(view, SimTime(0.0, 0), m_eager)

        cap_sensors = fresh()
        # warmup: seeds the Inertial Measurement Unit (IMU) finite-diff prev so capture runs with first=0
        for s in cap_sensors:
            s.sample_wp(view, SimTime(0.0, 0))
        with wp.ScopedCapture() as cap:
            for s in cap_sensors:
                s.sample_wp(view, SimTime(0.0, 0))
        wp.capture_launch(cap.graph)
        wp.synchronize()
        m_cap = Measurement()
        for s in cap_sensors:
            s.read(m_cap)

        # noise off + static view -> deterministic, step-independent; captured == eager
        assert (m_cap.xgyro, m_cap.ygyro, m_cap.zgyro) == pytest.approx(
            (m_eager.xgyro, m_eager.ygyro, m_eager.zgyro), abs=1e-6
        )
        assert (m_cap.xmag, m_cap.ymag, m_cap.zmag) == pytest.approx((m_eager.xmag, m_eager.ymag, m_eager.zmag))
        assert m_cap.abs_pressure == pytest.approx(m_eager.abs_pressure)
        assert (m_cap.lat_deg, m_cap.lon_deg, m_cap.alt_m) == pytest.approx(
            (m_eager.lat_deg, m_eager.lon_deg, m_eager.alt_m)
        )
