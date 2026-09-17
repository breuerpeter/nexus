# Copyright 2026, Freefly Systems. SPDX-License-Identifier: BSD-3-Clause
# Acronyms: Universal Scene Description (USD), Collective Thrust and Body Rates (CTBR).
"""GoTo task: fly a quadrotor to a sampled goal position and hold it, on the **single-body
per-rotor model**. Vehicle-agnostic: the robot + frame come from the cfg, and a concrete vehicle config lives
under ``config/``, for example ``config/astro_max.py``; the env reads the actuator thrust map from the
vehicle Universal Scene Description (USD) file.

The articulated route, real rotor joints driven by an IsaacLab ``DCMotor`` + a per-rotor external aero
wrench, is numerically intractable in the substep-integrated training env: ``num_substeps`` dilutes the
external rotor wrench, so correcting for that delivers an impulsive ×N drag kick to the tiny-inertia
rotor bodies → rotor-Ω jitter → an open-loop-unstable attitude loop that no RL config could get past, a
hard ~0.2 success wall. See ``scripts/probe_hover.py``.

So training uses a **single-body per-rotor model**: a single rigid body + per-rotor motor-speed states
with first-order lag, τ≈0.033 s, + a control-allocation matrix ``B``; forward ``B`` re-mixes the lagged,
saturated per-rotor thrusts into one base-body wrench. This keeps the real fidelity, motor lag + per-rotor
allocation + saturation/yaw limit, while being numerically stable; the motor-speed states also drive the
spinning-rotor visuals.

Deploy parity: the core runtime flies the same unified ``Rotors`` actuator, the same ``rigid_body_wrench``
kernel with ``dim = 1``, over the same mixer, so train and deploy are byte-shared, FR-7, the gate being the
policy-level transfer in ``nexus/examples/controllers/policy/goto/flight.py``.
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp
from isaaclab.utils.configclass import configclass

from nexus._src.physics.builders.usd import parse_rotor_joint_params  # motor:*/propeller:* from the USD joints
from nexus._src.vehicle.actuators import RPM_PER_RADS, quat_to_R  # shared rotor geometry, from the core
from nexus.examples._lib import (  # the shared single-body model + mixer: train + deploy + diff
    build_allocation,  # the one allocation builder; replaces the hand-rolled torch B
    ctbr_to_cmd_batched,  # the same CTBR mixer the deploy policy controller runs, with dim=N here
    motor_alpha,  # first-order motor lag α from (τ, dt)
    rigid_body_wrench_batched,  # the same unified motor model the deploy actuator runs, with dim=N here
)
from nexus.examples._lib.observation import (  # shared single source for train+deploy obs
    OBS_DIM,  # kinematic obs dim, 12
    POLICY_OBS_DIM,  # kinematic + last-action obs dim, 16
    observation_from_state,  # shared obs-from-state, the single source
)

from .quadcopter_newton import NUM_SUBSTEPS, QuadcopterNewtonEnv, QuadcopterNewtonEnvCfg


@configclass
class GoToEnvCfg(QuadcopterNewtonEnvCfg):
    """Generic GoTo per-rotor task cfg, vehicle-agnostic. A concrete vehicle config, ``config/<vehicle>.py``,
    fills in ``robot`` + ``base_body`` + ``thrust_sign`` and registers the task. Adds the per-rotor tweaks:
    the trained-policy obs carries the last action, ``POLICY_OBS_DIM=16``, to observe the lagged motor
    state, and larger MuJoCo-Warp constraint caps, since ground contacts early in training overflow the
    stock 40/20.
    """

    prev_action_obs: bool = True  # append the last action to the obs, to observe the lagged motor state
    njmax: int = 150  # MuJoCo-Warp per-world constraint cap; the stock 40 overflows on crashes early in training
    nconmax: int = 60  # matches the standalone deploy's njmax/nconmax
    motor_tau: float = 0.033  # motor first-order lag [s]

    def __post_init__(self):
        super().__post_init__()  # Newton MJWarp + kitless hygiene + CTBR params
        self.observation_space = POLICY_OBS_DIM if self.prev_action_obs else OBS_DIM
        self.sim.physics.solver_cfg.njmax = self.njmax
        self.sim.physics.solver_cfg.nconmax = self.nconmax


class GoToEnv(QuadcopterNewtonEnv):
    """Single-body per-rotor model: CTBR → inner rate loop → ``B⁻¹`` → per-rotor motor-speed
    states with first-order lag → forward ``B`` → one base-body wrench.
    Reuses the base env's CTBR params, obs including the last action, reward, NaN-robustness and
    start-state randomization; only the actuation differs: ``_pre_physics_step``, which writes the same
    ``self._thrust``/``self._moment`` the stock ``_apply_action`` applies to the base body. The env reads
    the thrust map, ct/cd/rpm_max, from the vehicle USD, the rotor-joint ``motor:*``/``propeller:*`` attrs,
    so it carries no per-vehicle constants.
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        robot = self._robot
        bn = list(robot.body_names)
        self._rotor_bodies = [i for i, n in enumerate(bn) if "rotor" in n.lower()]
        self._nr = len(self._rotor_bodies)
        # Actuator aero/thrust + motor map from the vehicle USD rotor joints: the one source the core also
        # reads, in physics/builders/usd.parse_rotor_joint_params, so no hardcoded per-vehicle dup.
        act = parse_rotor_joint_params(cfg.robot.spawn.usd_path)
        self._kf = act["ct"] * RPM_PER_RADS**2  # thrust = kf·Ω², with Ω in rad/s
        self._omega_max = act["rpm_max"] / RPM_PER_RADS  # motor saturation speed [rad/s]
        # First-order motor lag at the control rate: Ω += α·(Ω_cmd − Ω), α = 1 − e^{−Δt/τ}.
        # Δt is the control step, since the wrench holds over the decimation steps, so training,
        # Δt = step_dt, and deploy, Δt = sim dt, correctly get different α from the same τ. Shared helper.
        self._motor_alpha = motor_alpha(self.cfg.motor_tau, self.step_dt)
        self._omega = torch.zeros(self.num_envs, self._nr, device=self.device)  # motor-speed states, persistent

        # Allocation B, per-rotor thrust → [T, τx, τy, τz], from the rest-pose rotor geometry, via the one
        # shared builder, in numpy, the same B⁻¹/spins the deploy actuator builds. Uploaded to Warp arrays the
        # shared kernel reads; the torch refs keep the zero-copy storage alive.
        d = robot.data
        base_idx = int(self._body_id[0]) if hasattr(self._body_id, "__len__") else int(self._body_id)
        pos = d.body_pos_w[0].detach().cpu().numpy()  # (nbodies, 3)
        quat = d.body_quat_w[0].detach().cpu().numpy()  # (nbodies, 4), in x, y, z, w order, native Newton
        B, B_inv, _spins = build_allocation(pos, quat, base_idx, self._rotor_bodies, act["cd"])
        self._B_t = torch.tensor(B, dtype=torch.float32, device=self.device)  # (4, nr)
        self._B_inv_t = torch.tensor(B_inv, dtype=torch.float32, device=self.device)  # (nr, 4)
        self._B_wp = wp.from_torch(self._B_t, dtype=wp.float32)
        self._B_inv_wp = wp.from_torch(self._B_inv_t, dtype=wp.float32)
        self._cmd_wp = wp.zeros((self.num_envs, self._nr), dtype=float)  # (N, nr) per-rotor command, mixer → motor
        # Per-rotor base-frame offsets r, for the unified model's airflow inflow ω×r. astro-max authors
        # aero_h = aero_hforce = 0, so the inflow goes unused here, but it stays wired for the one shared kernel.
        Rb = quat_to_R(quat[base_idx])
        offsets = np.array([Rb.T @ (pos[rb] - pos[base_idx]) for rb in self._rotor_bodies], dtype=np.float32)
        self._aero_h, self._aero_hforce = float(act.get("aero_h", 0.0)), float(act.get("aero_hforce", 0.0))
        self._offsets_t = torch.tensor(offsets, dtype=torch.float32, device=self.device)  # (nr, 3)
        self._offsets_wp = wp.from_torch(self._offsets_t, dtype=wp.vec3)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)
        # Apply the same two ops the deploy seam runs, over zero-copy wp.from_torch views: the CTBR mixer,
        # ctbr_to_cmd_batched: inner rate loop → B⁻¹ → per-rotor command, then the single-body motor model,
        # rigid_body_wrench_batched: per-rotor command → motor lag/saturation → forward B → world wrench.
        # So the policy trains on the exact seam it deploys against, FR-7, byte-shared: the deploy policy
        # controller launches the same ctbr_to_cmd_batched mixer and the deploy actuator the same motor model,
        # with no torch reimplementation. substeps = NUM_SUBSTEPS carries the substep-dilution convention: the
        # base env's thrust_to_weight/rate_gain are the ×NUM_SUBSTEPS-baked twins, so the mixer's ÷substeps
        # recovers the effective T/W=1.9, gain=3, deploy's effective values, and the motor model's ×substeps
        # re-compensates the once-per-step dilution. The per-rotor motor-speed state, self._omega, updates in place.
        d = self._robot.data
        action_wp = wp.from_torch(self._actions.contiguous(), dtype=wp.vec4)
        quat_wp = wp.from_torch(d.root_quat_w.torch.contiguous(), dtype=wp.quat)  # native Newton, in x, y, z, w order
        omega_w_wp = wp.from_torch(d.root_ang_vel_w.torch.contiguous(), dtype=wp.vec3)  # world angular velocity
        # Base-body world twist, linear top / angular bottom, for the unified model's airflow inflow.
        twist_t = torch.cat([d.root_lin_vel_w.torch, d.root_ang_vel_w.torch], dim=-1).contiguous()
        twist_wp = wp.from_torch(twist_t, dtype=wp.spatial_vector)
        omega_state_wp = wp.from_torch(self._omega, dtype=wp.float32)  # (N, nr) motor-speed state, in/out
        # Outputs are writable views of self._thrust / self._moment, since (N,1,3)[:,0,:] is contiguous, so the
        # motor model writes the world-frame wrench straight into the tensors Isaac Lab's wrench composer reads.
        force_wp = wp.from_torch(self._thrust[:, 0, :], dtype=wp.vec3)
        torque_wp = wp.from_torch(self._moment[:, 0, :], dtype=wp.vec3)
        stream = wp.stream_from_torch() if action_wp.device.is_cuda else None  # the train-side torch↔Warp seam
        wp.launch(
            ctbr_to_cmd_batched,
            dim=self.num_envs,
            inputs=(
                action_wp,
                self._ctbr_params,
                quat_wp,
                omega_w_wp,
                self._B_inv_wp,
                self._kf,
                self._omega_max,
                float(NUM_SUBSTEPS),
                self._nr,
            ),
            outputs=(self._cmd_wp,),
            stream=stream,
        )
        wp.launch(
            rigid_body_wrench_batched,
            dim=self.num_envs,
            inputs=(
                self._cmd_wp,
                quat_wp,
                twist_wp,
                self._offsets_wp,
                self._B_wp,
                self._kf,
                self._omega_max,
                self._motor_alpha,
                float(NUM_SUBSTEPS),
                float(self.cfg.thrust_sign),
                self._aero_h,
                self._aero_hforce,
                self._nr,
                omega_state_wp,
            ),
            outputs=(force_wp, torque_wp),
            stream=stream,
        )
        # NB: no _apply_action override; the stock QuadcopterEnv._apply_action applies self._thrust /
        # self._moment to the base body, the same single-body path the lumped CTBR wrench uses.

    def _get_observations(self) -> dict:
        # The base obs, 12-D kinematic, body-frame, plus the policy's last applied action, so it can
        # infer the lagged motor state. The stock _reset_idx clears self._actions to 0 on reset, so the
        # first obs of an episode carries a zero last action, matching the deploy controller, which
        # appends its own last action the same way: FR-7 parity via the shared observation_from_state.
        d = self._robot.data
        obs = observation_from_state(
            d.root_pos_w.torch,
            d.root_quat_w.torch,
            d.root_lin_vel_w.torch,
            d.root_ang_vel_w.torch,
            self._desired_pos_w,
            prev_action=self._actions if self.cfg.prev_action_obs else None,
        )
        return {"policy": obs}

    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)  # base: stock reset, which zeros self._actions, + start-state randomization
        if env_ids is None or len(env_ids) == self.num_envs:
            import warp as wp

            env_ids = wp.to_torch(self._robot._ALL_INDICES)
        self._omega[env_ids] = 0.0  # rotors start from rest each episode
