"""The **coupling**: how the per-rotor motor + propeller wrenches reach the model. There is one
coupling: the kernel sums the per-rotor thrust + the ``κ·spin`` yaw + the airflow H-force into a single
base-body wrench, ``state.body_f[base]``, applied on both the collapsed single body and the articulated
multibody model. On the articulated model the rotor bodies hang force-free. For a rigid quad the summed
base wrench moves the rigid body exactly as the per-rotor wrenches would, and it's what the policy and
Proportional Integral Derivative (PID) deploy already do today. A multibody model's props spin for real
under the core ArticulatedRotors, through driven joints.

One ``@wp.func``, :func:`_summed_wrench`, composes the motor lag, :mod:`~nexus._src.vehicle.actuators.motor`,
+ the propeller, :mod:`~nexus._src.vehicle.actuators.propeller`, + the forward allocation ``B`` into the world
wrench. **Every** caller shares it through two thin kernels, and there is no per-consumer copy:

* :func:`rigid_body_wrench_world` writes ``body_f[base + n]``: the standalone deploy, the
  :class:`~nexus._src.vehicle.actuators.rotors.Rotors` actuator, ``dim = 1``; the differentiable
  design-opt rollout, ``dim = 1``, with distinct per-step Ω arrays for a clean tape; and the sampling
  Model Predictive Control (MPC) batched planner and real sim, ``dim = rollouts``.
* :func:`rigid_body_wrench_batched` writes separate ``out_force_w`` and ``out_torque_w`` outputs: the RL
  training env's zero-copy ``wp.from_torch`` views, the wrench Isaac Lab composes, ``dim = N``.

``substeps`` carries the Isaac-Lab substep convention: the kernel multiplies the final wrench by
``substeps``, and the mixer's ``ctbr_rate_loop`` divided the collective + rate-loop torque by it, so the
training twins and the deploy effective values produce the *same* effective per-rotor thrust.

The airflow inflow at each rotor is ``v_base + ω_base × r``, where ``r`` is the rotor's base-frame offset,
so the H-force and forward-flight terms stay correct after a collapse: they read the parsed layout, not
real rotor-body velocities. With ``aero_h = aero_hforce = 0`` the airflow terms are exact zeros and the
kernel matches the quasi-static ``kf·Ω²`` summed wrench bit for bit.
"""

from __future__ import annotations

import warp as wp

from nexus._src.vehicle.actuators.propeller import propeller_force
from nexus.examples._lib.motor import lag_step


@wp.func
def _summed_wrench(
    q: wp.quat,  # body→world rotation, xyzw order
    twist: wp.spatial_vector,  # base-body world twist: top = linear vel, bottom = angular vel
    offsets: wp.array(dtype=wp.vec3),  # (nr,) per-rotor base-frame offset r, for the airflow inflow ω×r
    thrust_sign: float,  # +1 = thrust along +body-z, −1 = Forward Right Down (FRD), −body-z
    B: wp.array2d(dtype=float),  # (4, nr) forward allocation: per-rotor thrust → [T, τx, τy, τz]
    cmd: wp.array2d(dtype=float),  # (·, nr) normalized per-rotor command u, the actuator's single input
    kf: float,  # thrust = kf·Ω², with Ω in rad/s
    omega_max_motor: float,  # rotor-speed saturation [rad/s]
    alpha: float,  # first-order motor lag at this path's control step
    substeps: float,  # Isaac-Lab substep factor: NUM_SUBSTEPS for train, 1 for deploy
    aero_h: float,  # forward-flight thrust-loss coeff, 0 = quasi-static kf·Ω²
    aero_hforce: float,  # in-plane rotor H-force coeff, 0 = none
    nr: int,  # rotor count
    n: int,  # row of the cmd and per-rotor state arrays: env or rollout, single-body 0
    omega_prev: wp.array2d(dtype=float),  # (·, nr) motor-speed state, input
    omega_next: wp.array2d(dtype=float),  # (·, nr) motor-speed state, output; can alias omega_prev for in-place
) -> wp.spatial_vector:
    """The unified motor + propeller + coupling: per-rotor command → motor lag → propeller, with airflow, →
    forward ``B`` → world wrench, ×substeps. ``Ω_cmd = clamp(u, 0, 1)·Ω_max`` makes the actuator the
    single saturation authority. Reads ``omega_prev[n]`` and writes the updated speeds into
    ``omega_next[n]``; pass the same array for both to update in place, as RL train and deploy do, or
    distinct per-step arrays for a clean differentiable recurrence, the design-opt and MPC tape.
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
    # differentiates cleanly, cosine 0.9996. Forward results are equal for the closed-loop callers, PID
    # and RL, up to float re-association. ×substeps compensates the once-per-step dilution; in deploy,
    # substeps = 1, a no-op.
    force_w = wp.vec3(0.0, 0.0, 0.0)
    torque_w = wp.vec3(0.0, 0.0, 0.0)
    for i in range(nr):
        omega_cmd = wp.clamp(cmd[n, i], 0.0, 1.0) * omega_max_motor  # u → rotor-speed target, saturated
        om = lag_step(omega_prev[n, i], omega_cmd, alpha, omega_max_motor)  # the motor lag, for fidelity
        omega_next[n, i] = om
        r = offsets[i]
        v_rot = v_base_b + wp.cross(w_base_b, r)  # rotor-hub inflow, body frame
        pf = propeller_force(om, v_rot[0], v_rot[1], kf, aero_h, aero_hforce)  # (h_x, h_y, thrust_mag)
        f_act = pf[2]  # axial thrust, with the forward-flight loss; == kf·Ω² at aero_h = 0
        h_x = pf[0]  # in-plane H-force, body x/y; zero at aero_hforce = 0, a no-op, quasi-static reduction
        h_y = pf[1]
        # this rotor's body-frame force, collective along body-z + in-plane H-force, and torque, roll/pitch/
        # yaw from B plus the H-force moment r × (h_x, h_y, 0); rotate each into world before summing.
        force_i = wp.vec3(h_x, h_y, thrust_sign * B[0, i] * f_act)
        torque_i = wp.vec3(
            B[1, i] * f_act - r[2] * h_y,
            B[2, i] * f_act + r[2] * h_x,
            B[3, i] * f_act + r[0] * h_y - r[1] * h_x,
        )
        force_w += wp.quat_rotate(q, force_i)
        torque_w += wp.quat_rotate(q, torque_i)

    return wp.spatial_vector(force_w * substeps, torque_w * substeps)


@wp.kernel
def rigid_body_wrench_batched(
    cmd: wp.array2d(dtype=float),  # (N, nr) normalized per-rotor command u
    quat: wp.array(dtype=wp.quat),  # (N,) body→world rotation, native-Newton xyzw order
    twist: wp.array(dtype=wp.spatial_vector),  # (N,) base-body world twist, linear top / angular bottom
    offsets: wp.array(dtype=wp.vec3),  # (nr,) per-rotor base-frame offset
    B: wp.array2d(dtype=float),  # (4, nr) forward allocation
    kf: float,
    omega_max_motor: float,
    alpha: float,
    substeps: float,
    thrust_sign: float,
    aero_h: float,
    aero_hforce: float,
    nr: int,
    omega_state: wp.array2d(dtype=float),  # (N, nr) per-rotor motor-speed state, persistent and in-place
    out_force_w: wp.array(dtype=wp.vec3),  # (N,) → world-frame force
    out_torque_w: wp.array(dtype=wp.vec3),  # (N,) → world-frame torque
):
    """Batched single-body wrench writing separate force/torque outputs, the RL training env's zero-copy
    ``wp.from_torch`` views, ``dim = N``: updates the per-rotor motor-speed state in place, with
    ``omega_state`` passed as both prev + next, and writes the world force/torque into the (N,) output
    views Isaac Lab reads.
    """
    n = wp.tid()
    sv = _summed_wrench(
        quat[n], twist[n], offsets, thrust_sign, B, cmd, kf, omega_max_motor, alpha, substeps,
        aero_h, aero_hforce, nr, n, omega_state, omega_state,
    )  # fmt: skip
    out_force_w[n] = wp.spatial_top(sv)
    out_torque_w[n] = wp.spatial_bottom(sv)


@wp.kernel
def rigid_body_wrench_world(
    cmd: wp.array2d(dtype=float),  # (N, nr) normalized per-rotor command u
    body_q: wp.array(dtype=wp.transform),  # per-body poses
    body_qd: wp.array(dtype=wp.spatial_vector),  # per-body twists, for the airflow inflow
    base: int,  # base/airframe body index of row 0; rows map to bodies base, base+1, …
    offsets: wp.array(dtype=wp.vec3),  # (nr,) per-rotor base-frame offset
    B: wp.array2d(dtype=float),  # (4, nr) forward allocation
    kf: float,
    omega_max_motor: float,
    alpha: float,
    substeps: float,
    thrust_sign: float,
    aero_h: float,
    aero_hforce: float,
    nr: int,
    omega_prev: wp.array2d(dtype=float),  # (N, nr) motor-speed state, input
    omega_next: wp.array2d(dtype=float),  # (N, nr) motor-speed state out; distinct per step ⇒ clean BPTT, or aliased
    out_body_f: wp.array(dtype=wp.spatial_vector),  # (nbodies,) per-body wrench; writes [base + n]
):
    """Single-body wrench writing the per-body wrench buffer: the standalone deploy, ``dim = 1``; the
    differentiable design-opt rollout, ``dim = 1`` with distinct Ω arrays; the sampling-MPC planner and real
    sim, ``dim = rollouts`` with one body per rollout. Each row ``n`` drives body ``base + n``.
    """
    n = wp.tid()
    bidx = base + n
    out_body_f[bidx] = _summed_wrench(
        wp.transform_get_rotation(body_q[bidx]), body_qd[bidx], offsets, thrust_sign, B, cmd, kf,
        omega_max_motor, alpha, substeps, aero_h, aero_hforce, nr, n, omega_prev, omega_next,
    )  # fmt: skip
