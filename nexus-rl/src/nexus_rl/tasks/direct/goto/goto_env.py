# Acronyms: Universal Scene Description (USD), Collective Thrust and Body Rates (CTBR).
"""GoTo task: fly the vehicle to a sampled goal position and hold it, on the plant every run flies.

Vehicle-agnostic: the robot and its frame come from the cfg, and a concrete vehicle config lives under
``config/``, for example ``config/astro_max.py``; the env reads the rotors and their values from the vehicle's
Universal Scene Description (USD) file.

The plant is the articulated vehicle: its rotor joints turn under the Newton motors the file authors, and each
rotor's propeller lifts its own body. Before each solver step, Isaac Lab's post-actuator hook runs
:meth:`GoToEnv._step_rotors`, which steps the two halves a run steps around its solver, in the same order: the
controller's CTBR mixer turns the held action and the airframe's body rate into each rotor's command, then the
rotor chain, :class:`~nexus_uav.examples._lib.rotor_chain.RotorChain`, sets each motor's speed target, adds each
propeller's force and steps the motors. Isaac Lab's own Newton actuator path stays off: it steps the motors
before the hook, a step before the command reaches them.

Deploy parity: the policy example flies the same mixer kernel, with the same values, over the same rotor chain
parts on the same plant; ``tests/examples/test_rl_plant_parity.py`` gates that the two answer one command alike.
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import NewtonManager

from nexus_uav.examples._lib import (  # the controller half and the plant a run flies
    RotorChain,  # the core's rotor chain, for every env's copy of the vehicle
    build_allocation,  # the one allocation builder the deploy mixer builds from too
    ctbr_to_cmd_batched,  # the same CTBR mixer the deploy policy controller runs, with dim=N here
)
from nexus_uav.examples._lib.observation import (  # shared single source for train+deploy obs
    OBS_DIM,  # kinematic obs dim, 12
    POLICY_OBS_DIM,  # kinematic + last-action obs dim, 16
    observation_from_state,  # shared obs-from-state, the single source
)
from nexus_uav.examples._lib.single_body import (  # the rotors the vehicle USD declares
    RPM_PER_RADS,
    parse_rotors,
)

from .quadcopter_newton import QuadcopterNewtonEnv, QuadcopterNewtonEnvCfg


@wp.kernel
def _airframe_attitude(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    bases: wp.array(dtype=wp.int32),  # (N,) each env's airframe body
    quat: wp.array(dtype=wp.quat),  # (N,) out: the airframe's attitude, x, y, z, w order
    omega_w: wp.array(dtype=wp.vec3),  # (N,) out: its world-frame angular velocity
):
    """Each env's airframe attitude and angular velocity, the CTBR mixer's state inputs, from the solver's state."""
    n = wp.tid()
    b = bases[n]
    quat[n] = wp.transform_get_rotation(body_q[b])
    omega_w[n] = wp.spatial_bottom(body_qd[b])


@configclass
class GoToEnvCfg(QuadcopterNewtonEnvCfg):
    """Generic GoTo task cfg, vehicle-agnostic. A concrete vehicle config, ``config/<vehicle>.py``, fills in
    ``robot`` + ``base_body`` + ``thrust_sign`` and registers the task. The trained-policy obs carries the last
    action, ``POLICY_OBS_DIM=16``, to observe the motors' lag behind it, and the MuJoCo-Warp constraint caps are
    larger, since ground contacts early in training overflow the stock 40/20.
    """

    prev_action_obs: bool = True  # append the last action to the obs, to observe the motors' lag
    njmax: int = 150  # MuJoCo-Warp per-world constraint cap; the stock 40 overflows on crashes early in training
    nconmax: int = 60  # matches the standalone deploy's njmax/nconmax

    def __post_init__(self):
        super().__post_init__()  # Newton MJWarp + kitless hygiene + CTBR params
        self.observation_space = POLICY_OBS_DIM if self.prev_action_obs else OBS_DIM
        self.sim.physics.solver_cfg.njmax = self.njmax
        self.sim.physics.solver_cfg.nconmax = self.nconmax


class GoToEnv(QuadcopterNewtonEnv):
    """The GoTo task on the articulated vehicle: CTBR action → inner rate loop → ``B⁻¹`` → each rotor's command →
    the rotor chain, before each solver step. Reuses the base env's CTBR params, obs including the last action,
    reward, NaN-robustness and start-state randomization. The env reads the rotors and their values, ct, cd and
    rpm_max, from the vehicle's USD file, so it carries no per-vehicle constants.
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        values, joints = parse_rotors(cfg.robot.spawn.usd_path)
        model = NewtonManager.get_model()
        self._chain = RotorChain(model, joints=joints, values=values)
        if self._chain.copies != self.num_envs:
            raise ValueError(f"the model holds {self._chain.copies} copies of the vehicle for {self.num_envs} envs")
        self._nr = self._chain.nr
        self._kf = float(values["ct"]) * RPM_PER_RADS**2  # thrust = kf·Ω², with Ω in rad/s
        self._omega_max = float(values["rpm_max"]) / RPM_PER_RADS  # rotor speed at full command [rad/s]
        # The mixer's B⁻¹, from the first env's rest-pose rotor geometry, through the one allocation builder the
        # deploy mixer builds from.
        body_q = NewtonManager.get_state_0().body_q.numpy()
        base, rotors = int(self._chain.rotor_bases[0, 0]), self._chain.rotor_bodies[0]
        _B, B_inv, _spins = build_allocation(body_q[:, :3], body_q[:, 3:7], base, rotors, float(values["cd"]))
        self._B_inv = wp.array(B_inv.astype(np.float32), dtype=float, device=self.device)
        n = self.num_envs
        # The action the hook holds over a policy step, a persistent buffer the mixer reads through a view.
        self._held_action = torch.zeros((n, 4), device=self.device)
        self._held_action_wp = wp.from_torch(self._held_action, dtype=wp.vec4)
        self._quat = wp.zeros(n, dtype=wp.quat, device=self.device)
        self._omega_w = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self._cmd = wp.zeros((n, self._nr), dtype=float, device=self.device)  # each rotor's command, mixer → chain
        self._dofs_per_env = model.joint_dof_count // n  # each env's degrees of freedom, contiguous
        NewtonManager.register_post_actuator_callback(self._step_rotors)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._held_action.copy_(self._actions)

    def _apply_action(self) -> None:
        """Nothing: the rotors lift the vehicle, through :meth:`_step_rotors` before each solver step."""

    def _step_rotors(self) -> None:
        """Before each solver step, as a run steps them: the CTBR mixer, from the held action and each airframe's
        body rate, then the rotor chain, from the mixer's commands.
        """
        state, control = NewtonManager.get_state_0(), NewtonManager.get_control()
        n = self.num_envs
        wp.launch(
            _airframe_attitude,
            dim=n,
            inputs=(state.body_q, state.body_qd, self._chain.bases),
            outputs=(self._quat, self._omega_w),
        )
        wp.launch(
            ctbr_to_cmd_batched,
            dim=n,
            inputs=(
                self._held_action_wp,
                self._ctbr_params,
                self._quat,
                self._omega_w,
                self._B_inv,
                self._kf,
                self._omega_max,
                self._nr,
            ),
            outputs=(self._cmd,),
        )
        self._chain.step(state, control, self._cmd, self.physics_dt)

    def _get_observations(self) -> dict:
        # The base obs, 12-D kinematic, body-frame, plus the policy's last applied action, so it can
        # infer the motors' lag. The stock _reset_idx clears self._actions to 0 on reset, so the
        # first obs of an episode carries a zero last action, matching the deploy controller, which
        # appends its own last action the same way, via the shared observation_from_state.
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
        super()._reset_idx(env_ids)  # base: stock reset, rotors at rest, + start-state randomization
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)
        dofs = torch.zeros((self.num_envs, self._dofs_per_env), dtype=torch.bool, device=self.device)
        dofs[env_ids] = True
        self._chain.reset(wp.from_torch(dofs.view(-1), dtype=wp.bool))  # the reset envs' motors start afresh
