"""Reusable Newton-backend base for Isaac Lab's quadcopter Direct task: the shared scaffolding the
registered GoTo task, ``goto_env.GoToEnv``, inherits. A clean **subclass** of the stock
``Isaac-Quadcopter-Direct-v0`` that works against an *unpatched* Isaac Lab checkout, so the multirotor RL
task trains kitless on standalone Newton, with no Isaac Sim / PhysX. This base carries the backend-portable
setup, the Collective Thrust and Body Rates (CTBR) params, and the NaN/reset robustness; ``GoToEnv``
supplies the **actuation**, ``_pre_physics_step``. It's **vehicle-agnostic**: the robot + frame, ``cfg.robot``
/ ``base_body`` / ``thrust_sign``, come from the cfg, set by a concrete vehicle config, ``config/<vehicle>.py``.

The stock task is PhysX-only: raw-torque actuation, ``root_view.get_masses()``. Making it
learn, reliably, on Newton needs these adaptations, all here so any vehicle inherits them:
  * **physics = Newton MJWarp**: the stock cfg ships no ``physics=`` preset -> defaults to PhysX;
    plus kitless/headless hygiene: ``debug_vis=False``, ``ui_window_class_type=None``,
    ``clone_in_fabric=False``, ``sim.visualizer_cfgs=[]``, which all otherwise pull in ``omni.kit``.
  * **CTBR action**, collective thrust + body rates, tracked by an inner proportional rate loop,
    instead of raw torque: stable and learnable on a backend with no angular-velocity clamp.
  * **world-frame wrench**: the Newton wrench composer applies forces in the world frame, so the
    env rotates the body-frame thrust/torque to world before applying them; otherwise the drone can
    change altitude but never translate, and never reaches the goal.
  * **NaN robustness**: a finite-state termination + reward sanitize so the inevitable early-
    training crashes can't poison rsl_rl's ``check_nan`` and end the run.
  * **start-state randomization**, ``_reset_idx``: the stock reset starts every env at the
    *same* default pose, so, with the stock ``entropy_coef=0``, a bad seed gets stuck hovering in
    place and never tracks the goal. Jittering the start pose + a velocity
    kick, paired with a small trainer entropy bonus, make ``success -> 1.0`` reliable across seeds.
  * backend-portable total mass via ``data.body_mass``, since Newton's ``ArticulationView`` has no
    PhysX ``get_masses()``.
"""

from __future__ import annotations

from dataclasses import MISSING

import gymnasium as gym
import isaaclab.sim as sim_utils
import torch
import warp as wp
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_tasks.direct.quadcopter.quadcopter_env import QuadcopterEnv
from isaaclab_tasks.direct.quadcopter.quadcopter_env_cfg import QuadcopterEnvCfg

from nexus.examples._lib import CtbrParams  # CTBR params struct for the per-rotor subclass's kernel
from nexus.examples._lib.observation import (  # shared single source for train+deploy
    observation_from_state,  # shared obs-from-state, the single source
)

NUM_SUBSTEPS = 4


@configclass
class QuadcopterNewtonEnvCfg(QuadcopterEnvCfg):
    """Quadcopter cfg on the Newton MJWarp backend, configured for kitless training."""

    # CTBR action interface: collective thrust + body rates tracked by an inner proportional rate loop.
    omega_max: float = 6.0  # max commanded body rate, rad/s
    rate_gain: float = 12.0  # inner rate-loop bandwidth, 1/s
    gyro_ff: bool = False  # ω×(Iω) gyroscopic feedforward, opt-in
    # Start-state randomization, the reliability fix; see _reset_idx. Modest magnitudes; the z jitter
    # stays well over the z<0.1 ground-death band.
    init_randomize: bool = True
    init_pos_xy: float = 0.5  # m, ± around the default start
    init_pos_z: float = 0.3  # m, ± around the default height
    init_lin_vel: float = 0.5  # m/s, ± per world axis
    init_ang_vel: float = 0.5  # rad/s, ± per body axis
    # Vehicle: supplied entirely by the concrete vehicle config, config/<vehicle>.py; the generic task
    # carries no vehicle, since the stock Crazyflie robot is overridden out, with no remnant. robot is the
    # ArticulationCfg; base_body / thrust_sign are its frame; the env reads the thrust map from its
    # Universal Scene Description (USD) file.
    robot: ArticulationCfg = MISSING
    base_body: str = MISSING  # base link the lumped thrust+moment applies to
    # which body-z is "up" toward the rotors: +z for Forward Left Up (FLU), -z for Forward Right Down (FRD)
    thrust_sign: float = MISSING

    def __post_init__(self):  # configclass runs this after field init
        self.debug_vis = False
        self.ui_window_class_type = None
        self.scene.clone_in_fabric = False
        self.sim = SimulationCfg(
            dt=1 / 100,
            render_interval=self.decimation,
            physics=NewtonCfg(
                solver_cfg=MJWarpSolverCfg(
                    njmax=40,
                    nconmax=20,
                    cone="pyramidal",
                    update_data_interval=2,
                    integrator="implicitfast",
                    impratio=1,
                ),
                num_substeps=NUM_SUBSTEPS,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        )
        self.sim.visualizer_cfgs = []  # headless: nothing for initialize_visualizers() to resolve
        # The env applies the external body wrench once per env-step, but the solver integrates it over
        # NUM_SUBSTEPS substeps, so NUM_SUBSTEPS dilutes its effective size; scale the collective
        # thrust up to keep the effective T/W = 1.9. The CTBR rate loop applies the same
        # NUM_SUBSTEPS factor to its torque; moment_scale goes unused, since attitude is rate-controlled.
        self.thrust_to_weight = 1.9 * NUM_SUBSTEPS


class QuadcopterNewtonEnv(QuadcopterEnv):
    """Quadcopter env with a backend-portable total-mass read, so it works on Newton. Vehicle-agnostic: the
    base link + thrust sign come from the cfg, ``base_body`` / ``thrust_sign``, the robot from
    ``cfg.robot``, so a new vehicle is a new config, not a new env class.
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        # Bypass QuadcopterEnv.__init__, whose root_view.get_masses() is PhysX-only and raises on
        # Newton, by running the grandparent setup and replicating the rest with data.body_mass.
        DirectRLEnv.__init__(self, cfg, render_mode, **kwargs)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._success_step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["lin_vel", "ang_vel", "distance_to_goal"]
        }
        self._body_id = self._robot.find_bodies(self.cfg.base_body)[0]
        self._robot_mass = self._robot.data.body_mass.torch[0].sum()  # backend-portable, unlike get_masses
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        # Base-body principal inertia, the diag of the 3x3, for the CTBR inner rate loop, torque = I*accel.
        # Reading it from the model makes the rate-loop gain inertia-independent, so it transfers from
        # the 28 g Crazyflie to the 9.2 kg Astro Max unchanged.
        inertia = self._robot.data.body_inertia.torch[0, self._body_id].reshape(-1, 9)[0]
        self._body_inertia_diag = inertia[[0, 4, 8]].to(self.device).view(1, 3)
        # CTBR params for the per-rotor subclass's shared mixer, ctbr_to_cmd_batched: the same mixer the
        # deploy policy controller runs, so training and deploy meet the same dynamics, byte for byte, with no
        # torch reimplementation, FR-7. Constants are the NUM_SUBSTEPS-diluted training twins of the deploy
        # values, thrust_to_weight=1.9*NUM_SUBSTEPS and _RATE_GAIN; the solver integrates the wrench over
        # NUM_SUBSTEPS substeps.
        idiag = self._body_inertia_diag.flatten().tolist()
        self._ctbr_params = CtbrParams()
        self._ctbr_params.thrust_to_weight = float(self.cfg.thrust_to_weight)
        self._ctbr_params.weight = float(self._robot_weight)
        self._ctbr_params.thrust_sign = float(self.cfg.thrust_sign)
        self._ctbr_params.omega_max = float(self.cfg.omega_max)
        self._ctbr_params.rate_gain = float(self.cfg.rate_gain)
        self._ctbr_params.inertia = wp.vec3(float(idiag[0]), float(idiag[1]), float(idiag[2]))
        self._ctbr_params.gyro_ff = 1.0 if self.cfg.gyro_ff else 0.0  # Layer 1, opt-in, default off
        self.set_debug_vis(self.cfg.debug_vis)

    # --- CTBR, collective-thrust + body-rate, action space ---
    # The stock task commands a raw body torque, moment_scale * action. On Newton that's both
    # unstable, since with no PhysX ang-vel clamp the low-inertia body spins to NaN, and hard to learn. The
    # proven quadrotor-RL interface is CTBR: action[1:4] is a target body rate,
    # tracked by an inner proportional rate loop whose error term IS the angular damping. This both
    # stabilizes the dynamics and gives the policy an easy interface, which is what lets it actually
    # learn to fly to the goal. The rate-loop params live on the cfg: omega_max / rate_gain / gyro_ff.
    #
    # The concrete subclass provides the actuation, ``_pre_physics_step``, writing the world-frame wrench
    # into self._thrust/self._moment: ``GoToEnv`` launches the single-body RigidBodyRotors kernel.
    # This Newton base only carries the backend-portable setup + CTBR params + NaN/reset robustness below.

    def _get_observations(self) -> dict:
        # Reuse the single shared obs-from-state builder, nexus.examples._lib.observation, the same
        # code the deploy controller/sensor calls, so the policy trains on observations that match those
        # it gets in the standalone runtime bit for bit: FR-7 / C-1 round-trip parity.
        # root_quat_w is native Newton, in x, y, z, w order, the convention observation_from_state expects.
        d = self._robot.data
        obs = observation_from_state(
            d.root_pos_w.torch,
            d.root_quat_w.torch,
            d.root_lin_vel_w.torch,
            d.root_ang_vel_w.torch,
            self._desired_pos_w,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # _get_rewards runs BEFORE the diverged env resets, since the DirectRLEnv step order is dones ->
        # rewards -> reset -> obs, so a transient solver NaN on a crash step would reach rsl_rl's
        # check_nan as a NaN reward. The finite check in _get_dones terminates those envs, and they
        # reset this same step, so sanitizing their about-to-be-discarded reward to finite is
        # correct, not masking: early training inevitably crashes drones into the ground.
        return torch.nan_to_num(super()._get_rewards(), nan=0.0, posinf=0.0, neginf=0.0)

    # --- Start-state randomization, the reliability fix ---
    # The stock _reset_idx puts every env at the same default pose and samples only the goal,
    # so with no exploration bonus a bad seed settles into hovering in place and
    # never learns to track the goal: full-length episodes, distance stuck ~1.4 m. Perturbing the
    # start, position, and especially a linear/angular velocity kick the policy must actively null,
    # forces it to learn goal-tracking + active stabilization from diverse states, which is what
    # makes convergence reliable across seeds. Vehicle-agnostic, so it lives in the base. Magnitudes
    # are deliberately modest, the z jitter stays well over the z<0.1 ground-death band, and live on
    # the cfg: init_randomize / init_pos_xy / init_pos_z / init_lin_vel / init_ang_vel.

    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)  # stock: logging, goal sample, writes default pose + zero vel
        if not self.cfg.init_randomize:
            return
        if env_ids is None or len(env_ids) == self.num_envs:
            import warp as wp

            env_ids = wp.to_torch(self._robot._ALL_INDICES)
        n = len(env_ids)
        dev = self.device
        # Reconstruct the start pose the stock reset just wrote, default pose + env origin, and jitter
        # its position only; attitude stays the convention-correct default to avoid quat-convention
        # pitfalls; the velocity kick already exercises recovery from off-level states.
        pose = self._robot.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += self._terrain.env_origins[env_ids]
        pose[:, 0] += torch.empty(n, device=dev).uniform_(-self.cfg.init_pos_xy, self.cfg.init_pos_xy)
        pose[:, 1] += torch.empty(n, device=dev).uniform_(-self.cfg.init_pos_xy, self.cfg.init_pos_xy)
        pose[:, 2] += torch.empty(n, device=dev).uniform_(-self.cfg.init_pos_z, self.cfg.init_pos_z)
        self._robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        vel = torch.empty(n, 6, device=dev)
        vel[:, 0:3].uniform_(-self.cfg.init_lin_vel, self.cfg.init_lin_vel)
        vel[:, 3:6].uniform_(-self.cfg.init_ang_vel, self.cfg.init_ang_vel)
        self._robot.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # NaN backstop: the stock task terminates only on an altitude band, and NaN<0.1 / NaN>2.0
        # are both False, so a diverged env would never reset before _get_observations() builds its
        # NaN obs and rsl_rl's check_nan crashes. The rate clamp described earlier prevents the spin-up;
        # this finite check is the belt-and-suspenders backstop that resets any env that still
        # went non-finite, to clean defaults, BEFORE the next observation build.
        died, time_out = super()._get_dones()
        d = self._robot.data
        diverged = (
            ~torch.isfinite(d.root_pos_w.torch).all(dim=1)
            | ~torch.isfinite(d.root_lin_vel_w.torch).all(dim=1)
            | ~torch.isfinite(d.root_ang_vel_w.torch).all(dim=1)
            | ~torch.isfinite(d.root_quat_w.torch).all(dim=1)
        )
        return died | diverged, time_out
