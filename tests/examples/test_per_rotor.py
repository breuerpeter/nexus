"""Parity + invariants for the shared single-body RigidBodyRotors model in nexus.examples._lib.

The Collective Thrust and Body Rates (CTBR) pipeline splits across the seam: the controller's mixer,
``ctbr_to_cmd_batched``: inner rate loop → ``B⁻¹`` → per-rotor command, feeds the motor model,
``rigid_body_wrench_batched``: per-rotor command → motor lag/saturation → forward ``B`` → wrench. These
tests pin that **the two composed reproduce the original all-in-one arithmetic**, the numpy golden, so the
split is byte-exact. They run model-free, with Warp on CPU and no torch / Isaac Lab needed, and lock the
design invariants the refactor relied on:

* **R1**: the motor-lag ``α`` derives from ``(τ, Δt)``, so train at Δt≈0.02 and deploy at Δt=0.004 get
  different ``α`` from the same ``τ``; a shared baked ``α`` would silently change training dynamics.
* **R2/substep equivalence**: with the baked training twins, ``T/W=1.9·N``, ``rate_gain=12`` and
  ``substeps=N``, the kernel produces the *same* per-rotor motor-speed state and ``N×`` the wrench of the
  deploy twins, ``T/W=1.9``, ``rate_gain=3`` and ``substeps=1``, so the same *effective* wrench,
  including under saturation: the synthesis's feared saturation divergence doesn't occur once the baked
  twins are in use.
* **R3**: the Warp kernel matches the numpy golden; the backend swap is exact to float tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from nexus.examples._lib import (
    CtbrParams,
    build_allocation,
    ctbr_to_cmd_batched,
    motor_alpha,
    rigid_body_wrench_batched,
)

wp.set_device("cpu")

RPM_PER_RADS = 60.0 / (2.0 * np.pi)
_CT, _CD, _RPM_MAX = 0.000003463, 0.05, 3800.0
_KF = _CT * RPM_PER_RADS**2
_OMEGA_MAX_MOTOR = _RPM_MAX / RPM_PER_RADS
_WEIGHT = 9.18 * 9.81
_INERTIA = np.array([0.143, 0.143, 0.259])

# A quad-X rest pose, base body 0 + 4 rotors, alternating cw/ccw via rotor-z, xyzw: identity=up, 180°x=down.
_POS = np.array([[0, 0, 0.0], [0.2, 0.2, 0], [-0.2, -0.2, 0], [0.2, -0.2, 0], [-0.2, 0.2, 0]], dtype=float)
_QUAT = np.array([[0, 0, 0, 1.0], [0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=float)


def _quat_to_R(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _params(*, t2w=1.9, rate_gain=3.0, gyro=False) -> CtbrParams:
    p = CtbrParams()
    p.thrust_to_weight = float(t2w)
    p.weight = float(_WEIGHT)
    p.thrust_sign = -1.0
    p.omega_max = 6.0
    p.rate_gain = float(rate_gain)
    p.inertia = wp.vec3(*(float(v) for v in _INERTIA))
    p.gyro_ff = 1.0 if gyro else 0.0
    return p


def _golden(action, quat, omega_w, B, B_inv, alpha, omega_state, *, t2w, rate_gain, substeps, gyro):
    """Exact numpy transcription of the *full* CTBR pipeline, mixer + motor model: the deploy path at
    substeps=1 *and* the training path at substeps=N with the /N…×N bookkeeping, the same code
    parameterized by substeps.
    """
    R = _quat_to_R(quat)
    omega_b = R.T @ omega_w
    a = np.clip(action, -1.0, 1.0)
    thr = (a[0] + 1.0) * 0.5
    thrust = (t2w / substeps) * _WEIGHT * thr
    g = rate_gain / substeps
    omega_des = a[1:4] * 6.0
    tau = _INERTIA * g * (omega_des - omega_b)
    if gyro:
        tau = tau + np.cross(omega_b, _INERTIA * omega_b)
    wrench_des = np.array([thrust, tau[0], tau[1], tau[2]])
    f_cmd = np.maximum(B_inv @ wrench_des, 0.0)
    omega_cmd = np.minimum(np.sqrt(f_cmd / _KF), _OMEGA_MAX_MOTOR)
    omega_state = omega_state + alpha * (omega_cmd - omega_state)
    omega_state = np.clip(omega_state, 0.0, _OMEGA_MAX_MOTOR)
    f_actual = _KF * omega_state**2
    wrench = B @ f_actual
    force_w = R @ np.array([0.0, 0.0, -1.0 * wrench[0]]) * substeps
    torque_w = R @ wrench[1:4] * substeps
    return force_w, torque_w, omega_state


def _run_kernel(actions, quats, omega_ws, B, B_inv, alpha, omega_state0, *, params, substeps):
    """Compose the seam, the CTBR mixer ctbr_to_cmd_batched then the motor model
    rigid_body_wrench_batched on CPU, and return (force_w, torque_w, omega_state) as numpy. The
    composition reproducing the all-in-one golden is the byte-exact-split proof.
    """
    n, nr = actions.shape[0], B.shape[1]
    action_wp = wp.array(actions.astype(np.float32), dtype=wp.vec4)
    quat_wp = wp.array(quats.astype(np.float32), dtype=wp.quat)
    omega_w_wp = wp.array(omega_ws.astype(np.float32), dtype=wp.vec3)
    B_wp = wp.array(B.astype(np.float32), dtype=float)
    B_inv_wp = wp.array(B_inv.astype(np.float32), dtype=float)
    omega_state = wp.array(omega_state0.astype(np.float32), dtype=float)
    cmd = wp.zeros((n, nr), dtype=float)  # the per-rotor command the mixer produces and the motor model reads
    twist = wp.zeros(n, dtype=wp.spatial_vector)  # base twist, the airflow inflow: zero + aero=0 ⇒ quasi-static
    offsets = wp.zeros(nr, dtype=wp.vec3)  # rotor offsets, unused at aero=0
    force_w = wp.zeros(n, dtype=wp.vec3)
    torque_w = wp.zeros(n, dtype=wp.vec3)
    wp.launch(
        ctbr_to_cmd_batched,
        dim=n,
        inputs=(action_wp, params, quat_wp, omega_w_wp, B_inv_wp, _KF, _OMEGA_MAX_MOTOR, float(substeps), nr),
        outputs=(cmd,),
    )
    wp.launch(
        rigid_body_wrench_batched,
        dim=n,
        inputs=(
            cmd,
            quat_wp,
            twist,
            offsets,
            B_wp,
            _KF,
            _OMEGA_MAX_MOTOR,
            float(alpha),
            float(substeps),
            params.thrust_sign,
            0.0,  # aero_h: airflow off ⇒ bit-for-bit the same as the quasi-static golden
            0.0,  # aero_hforce
            nr,
            omega_state,
        ),
        outputs=(force_w, torque_w),
    )
    wp.synchronize()
    return force_w.numpy(), torque_w.numpy(), omega_state.numpy()


def _rand_quat(rng, n):
    q = rng.standard_normal((n, 4))
    return (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(float)


def test_motor_alpha_dt_coupling():
    # R1: α = 1 − e^{−Δt/τ}, so train at Δt≈0.02 and deploy at Δt=0.004 differ from the same τ=0.033.
    assert motor_alpha(0.033, 0.02) == pytest.approx(0.454504, abs=1e-5)
    assert motor_alpha(0.033, 0.004) == pytest.approx(0.114154, abs=1e-5)
    assert motor_alpha(0.033, 0.004) < motor_alpha(0.033, 0.02)


def test_deploy_parity_vs_golden():
    """R3: the Warp kernel, deploy at substeps=1, matches the numpy golden over a random multi-step rollout,
    state recurrence included, across the unsaturated *and* saturated regimes.
    """
    rng = np.random.default_rng(0)
    B, B_inv, _ = build_allocation(_POS, _QUAT, 0, [1, 2, 3, 4], _CD)
    B32, B_inv32 = B.astype(np.float32).astype(float), B_inv.astype(np.float32).astype(float)  # isolate kernel arith
    n = 6
    alpha = motor_alpha(0.033, 0.004)
    p = _params()
    om_k = np.zeros((n, 4))
    om_g = np.zeros((n, 4))
    for _step in range(40):
        actions = rng.uniform(-1.2, 1.2, size=(n, 4))  # >1 exercises the input clamp + saturation
        quats = _rand_quat(rng, n)
        omega_ws = rng.uniform(-3, 3, size=(n, 3))
        fk, tk, om_k = _run_kernel(actions, quats, omega_ws, B32, B_inv32, alpha, om_k, params=p, substeps=1)
        fg = np.zeros((n, 3))
        tg = np.zeros((n, 3))
        for i in range(n):
            fg[i], tg[i], om_g[i] = _golden(
                actions[i],
                quats[i],
                omega_ws[i],
                B32,
                B_inv32,
                alpha,
                om_g[i],
                t2w=1.9,
                rate_gain=3.0,
                substeps=1,
                gyro=False,
            )
        np.testing.assert_allclose(om_k, om_g, rtol=2e-4, atol=1e-2)
        np.testing.assert_allclose(fk, fg, rtol=2e-4, atol=2e-3)
        np.testing.assert_allclose(tk, tg, rtol=2e-4, atol=2e-3)


def test_train_deploy_substep_equivalence():
    """R2: training at T/W=1.9·N, gain=12, substeps=N and deploy at T/W=1.9, gain=3, substeps=1 with the
    *same* α produce the *same* per-rotor motor-speed state and the same *effective* wrench, train =
    N×deploy, including under saturation, proving the /N…×N substep bookkeeping is exact.
    """
    rng = np.random.default_rng(1)
    N = 4
    B, B_inv, _ = build_allocation(_POS, _QUAT, 0, [1, 2, 3, 4], _CD)
    n = 8
    alpha = motor_alpha(0.033, 0.02)  # same α both paths, to isolate the substep bookkeeping
    p_deploy = _params(t2w=1.9, rate_gain=3.0)
    p_train = _params(t2w=1.9 * N, rate_gain=12.0)
    om_d = np.zeros((n, 4))
    om_t = np.zeros((n, 4))
    for _step in range(30):
        actions = rng.uniform(-1.5, 1.5, size=(n, 4))  # saturating regime
        quats = _rand_quat(rng, n)
        omega_ws = rng.uniform(-2, 2, size=(n, 3))
        fd, td, om_d = _run_kernel(actions, quats, omega_ws, B, B_inv, alpha, om_d, params=p_deploy, substeps=1)
        ft, tt, om_t = _run_kernel(actions, quats, omega_ws, B, B_inv, alpha, om_t, params=p_train, substeps=N)
        np.testing.assert_allclose(om_t, om_d, rtol=1e-5, atol=1e-4)  # same motor-speed states
        np.testing.assert_allclose(ft, fd * N, rtol=1e-5, atol=1e-4)  # train wrench = N × deploy, effective ==
        np.testing.assert_allclose(tt, td * N, rtol=1e-5, atol=1e-4)


def test_gyro_feedforward_layer():
    """The gyro_ff fidelity layer adds ω×(Iω) to the rate-loop torque; off by default = lumped baseline."""
    rng = np.random.default_rng(2)
    B, B_inv, _ = build_allocation(_POS, _QUAT, 0, [1, 2, 3, 4], _CD)
    alpha = motor_alpha(0.033, 0.004)
    actions = rng.uniform(-1, 1, size=(3, 4))
    quats = _rand_quat(rng, 3)
    omega_ws = rng.uniform(-4, 4, size=(3, 3))  # nonzero ω so the gyro term is active
    p = _params(gyro=True)
    _fk, tk, _ = _run_kernel(actions, quats, omega_ws, B, B_inv, alpha, np.zeros((3, 4)), params=p, substeps=1)
    for i in range(3):
        _fg, tg, _ = _golden(
            actions[i],
            quats[i],
            omega_ws[i],
            B,
            B_inv,
            alpha,
            np.zeros(4),
            t2w=1.9,
            rate_gain=3.0,
            substeps=1,
            gyro=True,
        )
        np.testing.assert_allclose(tk[i], tg, rtol=2e-4, atol=2e-3)


def test_saturation_clamps_omega():
    """An oversized collective command drives all rotors to the saturation speed Ω_max, the κ-limited yaw regime."""
    B, B_inv, _ = build_allocation(_POS, _QUAT, 0, [1, 2, 3, 4], _CD)
    alpha = 1.0  # no lag → reach the commanded, saturated, speed in one step
    actions = np.array([[1.0, 0.0, 0.0, 0.0]])  # max collective
    quats = np.array([[0.0, 0.0, 0.0, 1.0]])
    omega_ws = np.zeros((1, 3))
    p = _params(t2w=100.0)  # absurd thrust → omega_cmd far past Ω_max
    _f, _t, om = _run_kernel(actions, quats, omega_ws, B, B_inv, alpha, np.zeros((1, 4)), params=p, substeps=1)
    np.testing.assert_allclose(om[0], _OMEGA_MAX_MOTOR, rtol=1e-5)
