"""The **coupling** a planner keeps: how its single-body model's per-rotor motor + propeller wrenches reach the
body. Every run flies the articulated plant, its rotors turned by Newton's motors; a model-based controller that
plans on a lumped single body, the sampling Model Predictive Control (MPC) planner, sums each rotor's thrust, its
drag reaction, the reaction to its speed change and the airflow H-force into one base-body wrench,
``state.body_f[base + n]``, one body per rollout.

One ``@wp.func``, :func:`_summed_wrench`, composes the motor lag, :mod:`~nexus_uav.examples._lib.motor`,
+ the propeller, :mod:`~nexus_uav._src.vehicle.forces.propellers`, + the forward allocation ``B`` into the world
wrench, and :func:`rigid_body_wrench_world` launches it over the rollouts.

A rotor's motor turns it against the airframe, so the airframe takes the motor's torque back about the rotor's
axis: the steady part is the propeller drag, ``B``'s yaw row, and the part that changes the rotor's speed is
``I·dΩ/dt``, which the kernel adds from the rotor's spin inertia. On the articulated plant both come through the
rotor joint.

The airflow inflow at each rotor is ``v_base + ω_base × r``, where ``r`` is the rotor's base-frame offset,
so the H-force and forward-flight terms stay correct after a collapse: they read the parsed layout, not
real rotor-body velocities. With ``aero_h = aero_hforce = 0`` the airflow terms are exact zeros.
"""

from __future__ import annotations

import warp as wp

from nexus_uav._src.vehicle.forces.propellers import propeller_force
from nexus_uav.examples._lib.motor import lag_step


@wp.func
def _summed_wrench(
    q: wp.quat,  # body→world rotation, xyzw order
    twist: wp.spatial_vector,  # base-body world twist: top = linear vel, bottom = angular vel
    offsets: wp.array(dtype=wp.vec3),  # (nr,) per-rotor base-frame offset r, for the airflow inflow ω×r
    spins: wp.array(dtype=float),  # (nr,) each rotor's spin sign, the sign of its reaction on the airframe
    thrust_sign: float,  # +1 = thrust along +body-z, −1 = Forward Right Down (FRD), −body-z
    B: wp.array2d(dtype=float),  # (4, nr) forward allocation: per-rotor thrust → [T, τx, τy, τz]
    cmd: wp.array2d(dtype=float),  # (·, nr) normalized per-rotor command u
    kf: float,  # thrust = kf·Ω², with Ω in rad/s
    omega_max_motor: float,  # rotor-speed saturation [rad/s]
    alpha: float,  # first-order motor lag at this path's control step
    rotor_inertia: float,  # each rotor's spin inertia [kg·m²]
    step_dt: float,  # the step Ω changes over [s]
    aero_h: float,  # forward-flight thrust-loss coeff, 0 = quasi-static kf·Ω²
    aero_hforce: float,  # in-plane rotor H-force coeff, 0 = none
    nr: int,  # rotor count
    n: int,  # row of the cmd and per-rotor state arrays: the rollout
    omega_prev: wp.array2d(dtype=float),  # (·, nr) motor-speed state, input
    omega_next: wp.array2d(dtype=float),  # (·, nr) motor-speed state, output
) -> wp.spatial_vector:
    """The motor + propeller + coupling: per-rotor command → motor lag → propeller, with airflow, →
    forward ``B`` plus each rotor's speed-change reaction → world wrench. ``Ω_cmd = clamp(u, 0, 1)·Ω_max``.
    Reads ``omega_prev[n]`` and writes the updated speeds into ``omega_next[n]``, distinct per-step arrays for
    a clean differentiable recurrence.
    """
    # Base velocity in the body frame, for the per-rotor inflow v_rotor = v_base + ω_base × r.
    v_base_b = wp.quat_rotate_inv(q, wp.spatial_top(twist))
    w_base_b = wp.quat_rotate_inv(q, wp.spatial_bottom(twist))
    # Accumulate the wrench in the world frame: rotate each rotor's body-frame force/torque into world
    # *before* summing, with ``wp.vec3`` ``+=``, a differentiable ``add_inplace``. The alternative with the
    # same algebra, summing body-frame *scalar* accumulators such as ``t_x += …`` and ``wp.quat_rotate`` of
    # the single sum at the end, has a broken reverse-mode adjoint in Warp: ``for k in range(nr)`` with
    # ``nr`` a kernel arg is a dynamic loop, and mutating ``wp.float32`` scalars across it
    # violates Static Single Assignment (SSA), so Warp drops the loop-carried adjoint terms. The analytic
    # ∂cost/∂cmd comes out at ≈ −0.92 cosine compared to finite-difference: inverted, and ~10× too small.
    # The sampling-MPC planner runs Backpropagation Through Time (BPTT) on cost through this wrench to
    # refine its plans, so a mis-directed gradient makes the flight wander or spin; the world-frame form
    # differentiates cleanly, cosine 0.9996.
    force_w = wp.vec3(0.0, 0.0, 0.0)
    torque_w = wp.vec3(0.0, 0.0, 0.0)
    for i in range(nr):
        omega_cmd = wp.clamp(cmd[n, i], 0.0, 1.0) * omega_max_motor  # u → rotor-speed target, saturated
        om = lag_step(omega_prev[n, i], omega_cmd, alpha, omega_max_motor)  # the motor lag
        omega_next[n, i] = om
        r = offsets[i]
        v_rot = v_base_b + wp.cross(w_base_b, r)  # rotor-hub inflow, body frame
        pf = propeller_force(om, v_rot[0], v_rot[1], kf, aero_h, aero_hforce)  # (h_x, h_y, thrust_mag)
        f_act = pf[2]  # axial thrust, with the forward-flight loss; == kf·Ω² at aero_h = 0
        h_x = pf[0]  # in-plane H-force, body x/y; zero at aero_hforce = 0
        h_y = pf[1]
        # The motor's torque back on the airframe, the part that changed the rotor's speed this step.
        spin_up = spins[i] * rotor_inertia * (om - omega_prev[n, i]) / step_dt
        # this rotor's body-frame force, collective along body-z + in-plane H-force, and torque, roll/pitch/
        # yaw from B plus the H-force moment r × (h_x, h_y, 0) and the speed-change reaction; rotate each into
        # world before summing.
        force_i = wp.vec3(h_x, h_y, thrust_sign * B[0, i] * f_act)
        torque_i = wp.vec3(
            B[1, i] * f_act - r[2] * h_y,
            B[2, i] * f_act + r[2] * h_x,
            B[3, i] * f_act + spin_up + r[0] * h_y - r[1] * h_x,
        )
        force_w += wp.quat_rotate(q, force_i)
        torque_w += wp.quat_rotate(q, torque_i)

    return wp.spatial_vector(force_w, torque_w)


@wp.kernel
def rigid_body_wrench_world(
    cmd: wp.array2d(dtype=float),  # (N, nr) normalized per-rotor command u
    body_q: wp.array(dtype=wp.transform),  # per-body poses
    body_qd: wp.array(dtype=wp.spatial_vector),  # per-body twists, for the airflow inflow
    base: int,  # base/airframe body index of row 0; rows map to bodies base, base+1, …
    offsets: wp.array(dtype=wp.vec3),  # (nr,) per-rotor base-frame offset
    spins: wp.array(dtype=float),  # (nr,) each rotor's spin sign
    B: wp.array2d(dtype=float),  # (4, nr) forward allocation
    kf: float,
    omega_max_motor: float,
    alpha: float,
    rotor_inertia: float,
    step_dt: float,
    thrust_sign: float,
    aero_h: float,
    aero_hforce: float,
    nr: int,
    omega_prev: wp.array2d(dtype=float),  # (N, nr) motor-speed state, input
    omega_next: wp.array2d(dtype=float),  # (N, nr) motor-speed state out; distinct per step ⇒ clean BPTT
    out_body_f: wp.array(dtype=wp.spatial_vector),  # (nbodies,) per-body wrench; writes [base + n]
):
    """Single-body wrench writing the per-body wrench buffer: the sampling-MPC planner, ``dim = rollouts`` with
    one body per rollout. Each row ``n`` drives body ``base + n``.
    """
    n = wp.tid()
    bidx = base + n
    out_body_f[bidx] = _summed_wrench(
        wp.transform_get_rotation(body_q[bidx]), body_qd[bidx], offsets, spins, thrust_sign, B, cmd, kf,
        omega_max_motor, alpha, rotor_inertia, step_dt, aero_h, aero_hforce, nr, n, omega_prev, omega_next,
    )  # fmt: skip
