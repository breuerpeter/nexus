"""Captured execution strategy, architecture.md §5: the in-process device region replays from a
CUDA graph. Auto-skips without a CUDA device; restores the CPU device afterward so the bit-exact
determinism gate, which is CPU, stays unaffected.
"""

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("warp")

import warp as wp


def _stub_orch(*, phys_cap=True, act_cap=True, sens_cap=True, ctrl_host=False, ctrl_cap=False):
    """A minimal Orchestrator whose components carry only the capability markers
    _execution_strategy reads, so the routing is testable without a full sim build.
    """
    from nexus._src.core.orchestrator import Orchestrator

    def marker(**kw):
        return type("C", (), kw)()

    return Orchestrator(
        clock=marker(dt=0.004),
        environment=marker(),
        physics=marker(capturable=phys_cap),
        actuator=marker(capturable=act_cap),
        sensors=[marker(capturable=sens_cap)],
        controller=marker(host_boundary=ctrl_host, capturable=ctrl_cap),
    )


def test_execution_strategy_cpu_is_always_eager():
    """No CUDA device -> eager, regardless of the components' capturable markers, since capture needs CUDA."""
    try:
        wp.set_device("cpu")
        assert _stub_orch(ctrl_host=True)._execution_strategy() == "eager"
        assert _stub_orch(ctrl_cap=True)._execution_strategy() == "eager"
    finally:
        wp.set_device("cpu")


def test_execution_strategy_routes_on_cuda():
    """On CUDA the strategy captures whenever the device region is capturable; the controller only
    picks its seam: capturable in-process -> in the graph, captured-inprocess; anything else, a
    host-boundary peer or an in-process host solver -> the host seam, captured-host-exchange. Only
    a non-capturable device-region component forces eager.
    """
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    try:
        wp.set_device("cuda:0")
        assert _stub_orch(ctrl_host=True)._execution_strategy() == "captured-host-exchange"
        assert _stub_orch(ctrl_cap=True)._execution_strategy() == "captured-inprocess"
        # host_boundary wins over capturable; PX4 marks neither, but guard the precedence anyway
        assert _stub_orch(ctrl_host=True, ctrl_cap=True)._execution_strategy() == "captured-host-exchange"
        # any non-capturable device component, for example the Isaac physics, falls back to eager
        assert _stub_orch(phys_cap=False, ctrl_host=True)._execution_strategy() == "eager"
        assert _stub_orch(sens_cap=False, ctrl_cap=True)._execution_strategy() == "eager"
        # neither marker, an in-process host solver such as a Model Predictive Control (MPC), acados or
        # torch policy -> everything but the controller still captures; the exchange runs at the host seam
        assert _stub_orch()._execution_strategy() == "captured-host-exchange"
    finally:
        wp.set_device("cpu")


def test_host_rate_sensor_does_not_veto_capture():
    """A host_rate sensor, the RTX camera/lidar, host-bound and self-decimating, stays out of the capture
    gate and the graph partition: it must *not* force the eager strategy.
    """
    from nexus._src.core.orchestrator import Orchestrator

    def marker(**kw):
        return type("C", (), kw)()

    rtx = marker(host_rate=True)  # not capturable, but host-rate, so it doesn't veto
    imu = marker(capturable=True)
    orch = Orchestrator(
        clock=marker(dt=0.004),
        environment=marker(),
        physics=marker(capturable=True),
        actuator=marker(capturable=True),
        sensors=[imu, rtx],
        controller=marker(host_boundary=True, capturable=False),
    )
    assert orch._graph_sensors == [imu] and orch._host_sensors == [rtx]  # the partition
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    try:
        wp.set_device("cuda:0")
        assert orch._execution_strategy() == "captured-host-exchange"
    finally:
        wp.set_device("cpu")


def test_host_sensors_sampled_at_host_seam():
    """_sample_host fans the live state to every host-rate sensor; they self-decimate internally."""
    from nexus._src.core.orchestrator import Orchestrator

    def marker(**kw):
        return type("C", (), kw)()

    calls = []

    class Rtx:
        host_rate = True

        def sample(self, state, env, t, out):
            calls.append((state, t))

    orch = Orchestrator(
        clock=marker(dt=0.004),
        environment=marker(),
        physics=marker(capturable=True),
        actuator=marker(capturable=True),
        sensors=[Rtx()],
        controller=marker(host_boundary=True),
    )
    orch._sample_host("STATE", None, 0.5, None)
    assert calls == [("STATE", 0.5)]


def test_run_captured_pid_loop_executes():
    """The in-process Proportional Integral Derivative (PID) loop's per-tick device region captures +
    replays, producing a finite trajectory whose first step matches eager, before GPU contact-chaos
    diverges, per S-1.
    """
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    from nexus._src.config import LaunchConfig
    from nexus._src.runtimes.assembly import build_scenario, run_captured
    from nexus._src.runtimes.launch import resolve_to_vehicle_builder
    from nexus.examples.controllers.pid.assembly import build_pid_orchestrator

    def build(q_list, max_steps):
        cfg = build_scenario()
        cfg["physics"]["force_cpu"] = False
        vb, _ = resolve_to_vehicle_builder(LaunchConfig().set_vehicle("astro_max_base"))
        orch = build_pid_orchestrator(
            cfg,
            goal_w=(0.0, 0.0, 1.5),
            max_steps=max_steps,
            vehicle_builder=vb,
        )
        # Capture per-tick body_q via the post-step observer, which fires in the captured loop too; physics.state0
        # is the persistent state the graph advances in place.
        orch.on_tick = lambda view, t, n: q_list.append(orch.physics.state0.body_q.numpy().copy())
        return orch

    try:
        wp.set_device("cuda:0")
        q1 = []
        run_captured(build(q1, max_steps=None), steps=12)
        q = np.array(q1)
        assert q.shape[0] == 12 and np.isfinite(q).all()

        q2 = []
        build(q2, max_steps=2).run()  # eager, 2 steps: compare the first step, pre-chaos
        np.testing.assert_allclose(q[0], np.array(q2)[0], atol=1e-5)
    finally:
        wp.set_device("cpu")


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
    from nexus._src.core.schema import EnvSample, SimTime
    from nexus._src.core.seedtree import SeedTree
    from nexus._src.vehicle.sensors import ImuSensor

    try:
        wp.set_device("cuda:0")
        view = _WarpView((0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        env = EnvSample(gravity_world=(0.0, 0.0, -9.81), mag_ned=(0.21, 0.05, 0.43))
        s = ImuSensor(SeedTree(42), 0.004)  # noise on, the default
        s.sample_wp(view, env, SimTime(0.0, 0))  # warmup: first=1, seeds prev
        with wp.ScopedCapture() as cap:
            s.sample_wp(view, env, SimTime(0.0, 0))
        wp.capture_launch(cap.graph)
        wp.synchronize()
        a = s._out.numpy().copy()
        wp.capture_launch(cap.graph)
        wp.synchronize()
        b = s._out.numpy().copy()
        assert not np.allclose(a[:6], b[:6])  # acc+gyro noise dithers across replays
        np.testing.assert_allclose(a[6:], b[6:], atol=1e-6)  # quat stable, static state
    finally:
        wp.set_device("cpu")


def test_captured_px4_sensors_match_eager():
    """The Warp Measurement buffer, B: the Hardware In The Loop (HIL) sensors' kernels, sample_wp with no
    host readback, join a CUDA graph; replay + one read() reproduces the eager sample() values, so the
    PX4 device region is capturable. Validated with noise off so the result is step-independent.
    """
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    from nexus._src.core.schema import EnvSample, Measurement, SimTime
    from nexus._src.core.seedtree import SeedTree
    from nexus._src.vehicle.sensors import BaroSensor, GpsSensor, ImuSensor, MagSensor

    try:
        wp.set_device("cuda:0")
        view = _WarpView((1.0, -2.0, 3.0), (0.0, 0.0, 0.0, 1.0), (0.5, -0.1, 0.2), (0.3, -0.4, 0.5))
        env = EnvSample(gravity_world=(0.0, 0.0, -9.81), mag_ned=(0.21, 0.05, 0.43))

        def fresh():
            return [
                ImuSensor(SeedTree(42), 0.004, acc_noise=0.0, gyro_noise=0.0),
                MagSensor(SeedTree(42), noise=(0.0, 0.0, 0.0)),
                BaroSensor(SeedTree(42), noise=0.0),
                GpsSensor(47.6, -122.3, 5.0),
            ]

        eager = fresh()
        m_eager = Measurement()
        for s in eager:
            s.sample(view, env, SimTime(0.0, 0), m_eager)

        cap_sensors = fresh()
        # warmup: seeds the Inertial Measurement Unit (IMU) finite-diff prev so capture runs with first=0
        for s in cap_sensors:
            s.sample_wp(view, env, SimTime(0.0, 0))
        with wp.ScopedCapture() as cap:
            for s in cap_sensors:
                s.sample_wp(view, env, SimTime(0.0, 0))
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
    finally:
        wp.set_device("cpu")
