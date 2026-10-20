"""The **mixer**: the airframe control-allocation + Collective Thrust and Body Rates (CTBR) and moment
control laws that turn a high-level controller action into the per-rotor commands the rotors' ESCs
consume. This is the flight-controller half of the rotor chain: the law + the mixer live here and
in the controllers, and the motors are Newton's, on the rotor joints.

``Controls.command`` is **always** ``nr`` normalized per-rotor commands ``u ∈ [0, 1]``, the rotor-speed
fraction; the mixer is what produces them. Pieces, all ``B``-authoritative and shared by RL train + RL
deploy + the differentiable examples:

* :func:`build_allocation`: the one host-side builder of the allocation matrix ``B``, per-rotor thrust
  → ``[T, τx, τy, τz]``, and its pseudo-inverse ``B⁻¹``, the mixer core: wrench → per-rotor thrust, from
  the rest-pose rotor geometry. ``pinv`` is canonical and handles ``nr ≠ 4``.
* :func:`ctbr_rate_loop`: the CTBR → inner P rate-loop ``@wp.func`` returning the *target* base-frame
  wrench ``[T, τ]``, before allocation.
* :func:`wrench_to_cmd`: the shared mixer tail, ``B⁻¹`` → per-rotor thrust → normalized rotor-speed
  command ``u = √(max(f, 0)/kf) / Ω_max``. A clean differentiable allocation op, with no rate loop and no
  motor state, so design-opt's gradient flows through it.
* :func:`ctbr_to_cmd_batched` and :func:`moment_to_cmd_batched`: the two front-ends, CTBR rate loop and
  direct moments, feeding :func:`wrench_to_cmd`. The policy controller at deploy and the RL training env
  call ``ctbr_to_cmd_batched``; the Proportional Integral Derivative (PID) controller and design-opt call
  ``moment_to_cmd_batched``, so the CTBR→per-rotor mixer is byte-shared between train and deploy, the
  parity surface.
* :func:`build_rotor_mixer_from_model` and :func:`build_rotor_mixer_from_layout`: build the airframe
  :class:`RotorMixer`, the ``B``/``B⁻¹`` + thrust map, that configures the controller and a planner's coupling.

``CtbrParams`` lives here, not in ``controllers/policy``, because it's mixer configuration; homing it in
this leaf module keeps the import graph acyclic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from nexus_uav._src.vehicle.rotors import RPM_PER_RADS, quat_to_R, rotor_of, rotor_speeds


@dataclass(slots=True)
class Controls:
    """The controller→actuator command: always one command per actuator, normalized to ``[0, 1]``.

    The control law and the vehicle's mixer, the rate loop and the ``B⁻¹`` allocation, live in the
    controller, so whatever the law computes, Collective Thrust Body Rate (CTBR), moments or Nonlinear
    Model Predictive Control (NMPC) thrusts, ``command`` is what the controller's law emits and its stage
    writes to its controls signal: a length-``n`` host ``np.ndarray`` on the host exchange paths, or a
    device-native ``(1, n)`` Warp array in the captured loop.
    """

    command: Any = None  # one entry per actuator: an np.ndarray of length n, or a (1, n) Warp array


@wp.struct
class CtbrParams:
    thrust_to_weight: float
    weight: float  # m * g [N]
    thrust_sign: float  # +1 = Forward Left Up (FLU), thrust +body-z; −1 = Forward Right Down (FRD), −body-z
    omega_max: float  # rad/s: body-rate command scale (action[1:4] · omega_max)
    rate_gain: float  # 1/s: inner P rate-loop gain, commanded angular accel per rad/s of error
    inertia: wp.vec3  # base-body principal inertia diag [Ixx, Iyy, Izz]
    # --- fidelity layers: each default 0 = off = the lumped-CTBR baseline, byte for byte the same ---
    gyro_ff: float  # Layer 1: gyroscopic-feedforward gain; 1.0 adds ω×(Iω) to the rate-loop torque, 0 = off


def build_allocation(
    positions: np.ndarray,  # (nbodies, 3) world positions of every body
    quats: np.ndarray,  # (nbodies, 4) world orientations of every body, xyzw order
    base: int,  # base/airframe body index
    rotor_bodies,  # iterable of rotor body indices
    kappa,  # κ = cd, each rotor's reaction-torque/thrust ratio, for the yaw allocation: (nr,) or one for all
    thrust_sign: float = -1.0,  # +1 = FLU, thrust +body-z; −1 = FRD, −body-z: the collective's axis
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the control-allocation matrix ``B`` (4×nr), its mixer pseudo-inverse ``B⁻¹`` (nr×4), and the
    per-rotor spin signs, from the rest-pose geometry: the single source the three old ``_build_alloc``
    copies converge on. All in the **base-body frame**; ``B`` is rigid, rest-pose, built once.

    Rows of ``B``: ``[total thrust; roll τx; pitch τy; yaw τz (κ·spin)]``. The roll/pitch moment is the
    physical ``r × (thrust_sign·f·ẑ)``: the moment arms carry ``thrust_sign`` so the allocation is
    consistent with the collective force direction, FRD ⇒ thrust along −body-z. This matters for an
    external controller such as PX4: it computes per-motor outputs assuming the physical wrench, so ``B·f``
    must equal the wrench the rotors make. For the controllers whose law runs in nexus, ``B`` and ``B⁻¹``
    cancel, so the sign convention is internally immaterial, but getting it physically right is what lets
    a planner's single-body model match the plant. Each rotor's cw/ccw spin sign comes from its rotor-z
    compared to the body-z, as authored in the Universal Scene Description (USD).
    """
    Rb = quat_to_R(quats[base])
    bz_w = Rb @ np.array([0.0, 0.0, 1.0])
    pos_b, spins = [], []
    for rb in rotor_bodies:
        pos_b.append(Rb.T @ (positions[rb] - positions[base]))  # rotor offset in the base frame
        rz_w = quat_to_R(quats[rb]) @ np.array([0.0, 0.0, 1.0])
        spins.append(-np.sign(np.dot(rz_w, bz_w)))  # cw/ccw from rotor-z compared to body-z, USD-authored
    pos_b, spins = np.array(pos_b), np.array(spins)
    nr = len(spins)
    B = np.zeros((4, nr))
    B[0, :] = 1.0  # total thrust
    B[1, :] = thrust_sign * pos_b[:, 1]  # roll  τx = (r × thrust_sign·f·ẑ)_x
    B[2, :] = thrust_sign * (-pos_b[:, 0])  # pitch τy = (r × thrust_sign·f·ẑ)_y
    B[3, :] = np.asarray(kappa, dtype=np.float64) * spins  # yaw   τz, differential reaction drag
    B_inv = np.linalg.pinv(B)  # mixer, wrench → per-rotor thrust; pinv handles nr ≠ 4 generically
    return B, B_inv, spins


@wp.func
def ctbr_rate_loop(action: wp.vec4, omega_b: wp.vec3, p: CtbrParams) -> wp.vec4:
    """CTBR collective thrust + inner P rate-loop → the **target base-frame wrench** ``[T, τx, τy, τz]``,
    before the allocation, as a ``wp.vec4``. The single source of the per-rotor CTBR law, shared by the
    deploy and the RL training launches.
    """
    thr = (wp.clamp(action[0], -1.0, 1.0) + 1.0) * 0.5
    thrust = p.thrust_to_weight * p.weight * thr  # [N], the collective; sign applied later
    g = p.rate_gain
    omega_des = (
        wp.vec3(wp.clamp(action[1], -1.0, 1.0), wp.clamp(action[2], -1.0, 1.0), wp.clamp(action[3], -1.0, 1.0))
        * p.omega_max
    )
    d = omega_des - omega_b
    # τ = (I · g) · (ω_des − ω), left-associated to match the torch training env's I·(gain)·Δω order.
    tau = wp.vec3(p.inertia[0] * g * d[0], p.inertia[1] * g * d[1], p.inertia[2] * g * d[2])
    # Layer 1: gyroscopic feedforward ω×(Iω). Default off; gyro_ff = 1 ⇒ active, matches the deploy option.
    tau = tau + p.gyro_ff * wp.cross(omega_b, wp.cw_mul(p.inertia, omega_b))
    return wp.vec4(thrust, tau[0], tau[1], tau[2])


@wp.func
def wrench_to_cmd(
    wrench_des: wp.vec4,  # target base-frame wrench [T, τx, τy, τz]
    B_inv: wp.array2d(dtype=float),  # (nr, 4) mixer: wrench → per-rotor thrust
    kf: wp.array(dtype=float),  # (nr,) each rotor's thrust = kf·Ω², with Ω in rad/s
    omega_max_motor: wp.array(dtype=float),  # (nr,) each rotor's speed saturation [rad/s]
    nr: int,
    n: int,  # output row: env or rollout, single-body 0
    out_cmd: wp.array2d(dtype=float),  # (·, nr) → normalized per-rotor command u
):
    """The shared mixer tail: target wrench → ``B⁻¹`` → per-rotor thrust → **normalized rotor-speed
    command** ``u = √(max(f, 0)/kf) / Ω_max``. The rotors' command element clamps ``u`` to
    ``[0, 1]``, the single saturation authority, and the motors bring the rotors to speed, so this op carries only
    the differentiable allocation + thrust-map inverse, with no rate loop and no motor state, keeping a
    clean full-horizon gradient for the moment-input design-opt path. Splitting it out of the old single
    kernel is exact: ``clamp(u, 0, 1)·Ω_max = min(√(f/kf), Ω_max)`` reproduces the old in-kernel saturation.
    """
    for i in range(nr):
        f = B_inv[i, 0] * wrench_des[0] + B_inv[i, 1] * wrench_des[1]
        f = f + B_inv[i, 2] * wrench_des[2] + B_inv[i, 3] * wrench_des[3]
        f = wp.max(f, 0.0)  # per-rotor thrusts are non-negative
        out_cmd[n, i] = wp.sqrt(f / kf[i]) / omega_max_motor[i]  # rotor-speed fraction; the rotor's ESC clamps it


@wp.kernel
def ctbr_to_cmd_batched(
    action: wp.array(dtype=wp.vec4),  # (N,) CTBR command [collective, ωx, ωy, ωz], clamped to [-1, 1]
    p: CtbrParams,
    quat: wp.array(dtype=wp.quat),  # (N,) body→world rotation, native-Newton xyzw order
    omega_w: wp.array(dtype=wp.vec3),  # (N,) world-frame angular velocity
    B_inv: wp.array2d(dtype=float),  # (nr, 4) mixer
    kf: wp.array(dtype=float),  # (nr,) each rotor's thrust map
    omega_max_motor: wp.array(dtype=float),  # (nr,) each rotor's speed saturation [rad/s]
    nr: int,
    out_cmd: wp.array2d(dtype=float),  # (N, nr) → normalized per-rotor command
):
    """The CTBR mixer, the policy controller's airframe mixer: collective + inner P rate loop → ``B⁻¹``
    → per-rotor command. The same op the RL training env launches, ``dim = N``, and the deploy policy
    controller launches, ``dim = 1``: the byte-shared train↔deploy mixer. The kernel computes ``omega_b``
    from ``quat``/``omega_w``; deploy passes identity quat + the body-frame rate, recovering ``omega_b``.
    """
    n = wp.tid()
    omega_b = wp.quat_rotate_inv(quat[n], omega_w[n])  # world ω → body
    wrench_des = ctbr_rate_loop(action[n], omega_b, p)  # [T, τx, τy, τz] target
    wrench_to_cmd(wrench_des, B_inv, kf, omega_max_motor, nr, n, out_cmd)


@wp.kernel
def moment_to_cmd_batched(
    action: wp.array(dtype=wp.vec4),  # (N,) [collective, m_x, m_y, m_z]: direct moments, the PID/design-opt law
    thrust_to_weight: float,
    weight: float,  # m · g [N]
    moment_scale: float,  # scales the moment command [N·m per unit action]
    B_inv: wp.array2d(dtype=float),  # (nr, 4) mixer
    kf: wp.array(dtype=float),  # (nr,) each rotor's thrust map
    omega_max_motor: wp.array(dtype=float),  # (nr,) each rotor's speed saturation [rad/s]
    nr: int,
    out_cmd: wp.array2d(dtype=float),  # (N, nr) → normalized per-rotor command
):
    """The **moment** mixer, the PID and design-opt airframe mixer: the controller's collective + direct
    moments ``[collective, m_x, m_y, m_z]`` → ``B⁻¹`` → per-rotor command, **no CTBR rate loop**. The rate
    loop's ``ω_des − ω`` feedback is what pushes the closed-loop Backpropagation Through Time (BPTT)
    spectral radius >1; skipping it keeps a clean full-horizon gradient, and design-opt's leaf is the
    controller gains feeding this op.
    """
    n = wp.tid()
    a4 = action[n]
    thrust = thrust_to_weight * weight * (wp.clamp(a4[0], -1.0, 1.0) + 1.0) * 0.5  # collective [N]
    wrench_des = wp.vec4(thrust, moment_scale * a4[1], moment_scale * a4[2], moment_scale * a4[3])
    wrench_to_cmd(wrench_des, B_inv, kf, omega_max_motor, nr, n, out_cmd)


@wp.kernel
def pack_vec4(a: wp.array(dtype=float), out: wp.array(dtype=wp.vec4)):
    """Pack a length-4 ``[collective, m_x, m_y, m_z]`` or ``[collective, ωx, ωy, ωz]`` float action into the
    ``wp.vec4`` the mixer kernels read: a device-native op so the mixer stays graph-safe and tape-able.
    """
    out[0] = wp.vec4(a[0], a[1], a[2], a[3])


@dataclass
class RotorMixer:
    """The airframe mixer/coupling config, built once from the rotor geometry + the rotors' thrust map,
    used by the controller, which takes the mixer side, ``B_inv`` + the thrust map, and by a planner's single-body
    model, which takes the coupling side, forward ``B`` + the thrust map. Both derive from the same
    :func:`build_allocation`, so they can't drift.
    """

    B: np.ndarray  # (4, nr) forward allocation, a planner's body coupling
    B_inv: np.ndarray  # (nr, 4) mixer, the controller's allocation
    spins: np.ndarray  # (nr,) cw/ccw sign per rotor, the sign of its reaction on the airframe
    kf: np.ndarray  # (nr,) each rotor's thrust = kf·Ω², with Ω in rad/s
    omega_max_motor: np.ndarray  # (nr,) each rotor's speed saturation [rad/s]
    kappa: np.ndarray  # (nr,) each rotor's κ = cd, the yaw reaction-torque / thrust ratio
    nr: int  # rotor count
    base: int  # base/airframe body index, where the summed wrench acts
    rotor_offsets: np.ndarray  # (nr, 3) per-rotor base-frame offsets r, the airflow inflow ω×r in the coupling

    def thrust_to_weight(self, mass: float) -> float:
        """The plant's true thrust-to-weight for a vehicle of ``mass``: Σ kf·Ω_max² / (m·g) over the rotors,
        every term from the USD-authored thrust map. The controller-facing action scale should be this, so that
        action = +1 means the plant's real max thrust, not a declared constant.
        """
        return math.fsum(self.kf * self.omega_max_motor**2) / (float(mass) * 9.81)  # the exact sum, rounded once


def _thrust_map(ct, rpm_max, cd) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each rotor's ``kf``, ``Ω_max`` [rad/s] and ``κ`` from its ``ct`` [N/rpm²], ``rpm_max`` [rpm] and ``cd``."""
    return (
        np.asarray(ct, dtype=np.float64) * RPM_PER_RADS**2,
        np.asarray(rpm_max, dtype=np.float64) / RPM_PER_RADS,
        np.asarray(cd, dtype=np.float64),
    )


def build_rotor_mixer_from_model(model, rotors, rest_body_q) -> RotorMixer:
    """Build the :class:`RotorMixer` from a finalized model's rotors + each rotor's thrust map:
    the airframe that configures the controller, ``B_inv``, and forward ``B``. ``rotors`` are the rotors the
    vehicle declares in channel order, the ``VehicleUsd``'s ``rotors()``, so the mixer's columns are the
    controls' entries. ``rest_body_q`` is the settled rest pose ``(nbodies, 7)`` the allocation builds
    from, since the rest-pose geometry is rigid.
    """
    found = [rotor_of(model, rotor.body) for rotor in rotors]  # each rotor's dof, coord, body and airframe
    bodies, base = [body for _dof, _coord, body, _base in found], found[0][3]
    kf, omega_max, kappa = _thrust_map(
        [r.propeller["ct"] for r in rotors], rotor_speeds(model, rotors), [r.propeller["cd"] for r in rotors]
    )
    bq = np.asarray(rest_body_q, dtype=np.float32)
    B, B_inv, spins = build_allocation(bq[:, :3], bq[:, 3:7], base, bodies, kappa)
    Rb = quat_to_R(bq[base, 3:7])
    offsets = np.array([Rb.T @ (bq[rb, :3] - bq[base, :3]) for rb in bodies], dtype=np.float64)  # base-frame
    return RotorMixer(
        B=B,
        B_inv=B_inv,
        spins=spins,
        kf=kf,
        omega_max_motor=omega_max,
        kappa=kappa,
        nr=len(bodies),
        base=int(base),
        rotor_offsets=offsets,
    )


def build_rotor_mixer_from_layout(rotor_offsets, turning_dirs, *, ct, cd, rpm_max, thrust_sign: float = -1.0) -> RotorMixer:  # fmt: skip
    """Build the :class:`RotorMixer` from an explicit rotor layout, base-frame offsets + cw/ccw spins, and each
    rotor's ``ct``, ``cd`` and ``rpm_max``: the single-body proxy path, since the sampling Model Predictive
    Control (MPC) example has no model rotor joints. Forward ``B`` follows the same physical,
    ``thrust_sign``-carrying convention as :func:`build_allocation`. ``base = 0``: each proxy body is its own
    airframe.
    """
    off = np.asarray(rotor_offsets, dtype=np.float64)
    dirs = np.asarray(turning_dirs, dtype=np.float64)
    kf, omega_max, kappa = _thrust_map(ct, rpm_max, cd)
    nr = len(dirs)
    B = np.zeros((4, nr))
    B[0, :] = 1.0  # total thrust
    B[1, :] = thrust_sign * off[:, 1]  # roll  τx = (r × thrust_sign·f·ẑ)_x
    B[2, :] = thrust_sign * (-off[:, 0])  # pitch τy = (r × thrust_sign·f·ẑ)_y
    B[3, :] = kappa * dirs  # yaw  τz, differential reaction drag
    return RotorMixer(
        B=B,
        B_inv=np.linalg.pinv(B),
        spins=dirs,
        kf=kf,
        omega_max_motor=omega_max,
        kappa=kappa,
        nr=nr,
        base=0,
        rotor_offsets=off,
    )


class CtbrMixer:
    """The CTBR mixer a deploy policy controller runs each tick: CTBR action ``[collective, ωx, ωy, ωz]``
    + the body-frame rate → ``nr`` per-rotor commands. Device-native, launching :func:`ctbr_to_cmd_batched`
    with ``dim = 1``, over persistent buffers, so it joins a CUDA graph; it's the deploy half of the
    byte-shared train↔deploy mixer, and the training env launches the same kernel, ``dim = N``.

    The kernel computes ``omega_b = Rᵀ(q)·omega_w``; at deploy the controller already holds the body-frame
    rate, ``obs[3:6]``, so it passes the identity quaternion + that rate as ``omega_w``, recovering
    ``omega_b`` unchanged with the same op the training launch uses, with no extra state to plumb.
    """

    def __init__(self, mixer: RotorMixer, ctbr_params: CtbrParams):
        self.nr = int(mixer.nr)
        self.p = ctbr_params
        self.kf = wp.array(np.asarray(mixer.kf, dtype=np.float32), dtype=float)
        self.omega_max_motor = wp.array(np.asarray(mixer.omega_max_motor, dtype=np.float32), dtype=float)
        self._B_inv = wp.array(np.asarray(mixer.B_inv, dtype=np.float32), dtype=float)
        self._action = wp.zeros(1, dtype=wp.vec4)
        self._quat = wp.zeros(1, dtype=wp.quat)
        self._omega_w = wp.zeros(1, dtype=wp.vec3)
        self._cmd = wp.zeros((1, self.nr), dtype=float)
        self._identity = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)  # identity, xyzw order

    def cmd(self, action4, omega_b) -> wp.array:
        """Run the mixer: ``(action4, omega_b)``, numpy length-4 and length-3, → the persistent ``(1, nr)``
        per-rotor command Warp array. Identity quat + ``omega_b`` as ``omega_w`` ⇒ ``omega_b`` recovered.
        """
        self._action.assign(np.asarray(action4, dtype=np.float32).reshape(1, 4))
        self._quat.assign(self._identity)
        self._omega_w.assign(np.asarray(omega_b, dtype=np.float32).reshape(1, 3))
        wp.launch(
            ctbr_to_cmd_batched,
            dim=1,
            inputs=(
                self._action,
                self.p,
                self._quat,
                self._omega_w,
                self._B_inv,
                self.kf,
                self.omega_max_motor,
                self.nr,
            ),
            outputs=(self._cmd,),
        )
        return self._cmd


class MomentMixer:
    """The moment mixer a deploy PID controller runs each tick: a direct-moment action
    ``[collective, m_x, m_y, m_z]``, a Warp ``(4,)`` array, the device-native ``pid_law`` output, → ``nr``
    per-rotor commands. Device-native, ``pack_vec4`` → :func:`moment_to_cmd_batched` with ``dim = 1``, over
    persistent buffers, so the PID loop captures into a CUDA graph.
    """

    def __init__(self, mixer: RotorMixer, *, thrust_to_weight: float, weight: float, moment_scale: float):
        self.nr = int(mixer.nr)
        self.thrust_to_weight = float(thrust_to_weight)
        self.weight = float(weight)  # m · g [N]
        self.moment_scale = float(moment_scale)
        self.kf = wp.array(np.asarray(mixer.kf, dtype=np.float32), dtype=float)
        self.omega_max_motor = wp.array(np.asarray(mixer.omega_max_motor, dtype=np.float32), dtype=float)
        self._B_inv = wp.array(np.asarray(mixer.B_inv, dtype=np.float32), dtype=float)
        self._act4 = wp.zeros(1, dtype=wp.vec4)
        self._cmd = wp.zeros((1, self.nr), dtype=float)

    def cmd_wp(self, moments_wp, out=None) -> wp.array:
        """Run the mixer over a device-native ``(4,)`` Warp moment action → the persistent ``(1, nr)``
        per-rotor command Warp array, graph-safe, with no host hop: ``out`` when given, such as the
        controls signal's buffer, else the mixer's own.
        """
        wp.launch(pack_vec4, dim=1, inputs=(moments_wp,), outputs=(self._act4,))
        wp.launch(
            moment_to_cmd_batched,
            dim=1,
            inputs=(
                self._act4,
                self.thrust_to_weight,
                self.weight,
                self.moment_scale,
                self._B_inv,
                self.kf,
                self.omega_max_motor,
                self.nr,
            ),
            outputs=(self._cmd if out is None else out,),
        )
        return self._cmd if out is None else out
