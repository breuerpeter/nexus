"""Astro Max: the concrete vehicle config for the GoTo task, in the Isaac Lab task/config convention.

Supplies the robot, the astro-max Universal Scene Description (USD) ArticulationCfg, and its frame, the base
link ``body_frd`` and the Forward Right Down (FRD) thrust sign, to the vehicle-agnostic
:class:`~nexus_rl.tasks.direct.goto.goto_env.GoToEnv`, and registers the task
``Newton-AstroMax-GoTo-Direct-v0``. A different vehicle is just another ``config/<vehicle>.py``, with no env
change. The env reads the thrust map, ct/cd/rpm_max, from the USD; it isn't set here.

The USD is a content-addressed registry asset, served via CloudFront; pass ``--vehicle_usd`` to the
train/play scripts to override the resolved path. They mutate ``env_cfg.robot.spawn.usd_path`` after
loading the registry cfg, the idiomatic Isaac Lab override.
"""

from __future__ import annotations

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.configclass import configclass

from ..goto_env import GoToEnvCfg

_TASK = "nexus_rl.tasks.direct.goto"


def _resolve_astro_max_usd() -> str:
    """Resolve the Astro Max USD for Isaac Lab to load: the **hosted, content-addressed registry
    asset**, the same source the standalone runtime resolves; cached, so offline after the first fetch.
    """
    from nexus._src.config import LaunchConfig, resolve

    # resolve() fetches with a bounded timeout, via newton-assets; a sha-mismatch raises ValueError so an
    # integrity failure surfaces loudly instead of silently using a stale local file.
    return str(resolve(LaunchConfig().set_vehicle("astro_max_base")).vehicle_usd_path)


ASTRO_MAX_USD = _resolve_astro_max_usd()


def _spawn_astro_max_freeflying(prim_path, cfg, translation=None, orientation=None):
    """Load the Astro Max USD into the stage, then free its base. The asset bakes an explicit
    ``PhysicsFixedJoint`` at ``/astro_max/Physics/root_joint``, world Xform -> ``body_frd``, that pins the
    base, so Isaac Lab loads it ``is_fixed_base=True`` and thrust/torque move nothing.
    ``ArticulationRootPropertiesCfg(fix_root_link=False)`` can't remove it: Isaac Lab only detects
    single-target anchor joints, and this one names both bodies. So this function turns the joint off on
    the loaded prim; the standalone runtime achieves the same with ``add_usd(floating=True)``.
    """
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    from pxr import UsdPhysics

    prim = spawn_from_usd(prim_path, cfg, translation, orientation)
    joint = prim.GetStage().GetPrimAtPath(f"{prim.GetPath()}/Physics/root_joint")
    if joint.IsValid():
        UsdPhysics.Joint(joint).GetJointEnabledAttr().Set(False)
    return prim


ASTRO_MAX_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ASTRO_MAX_USD,
        func=_spawn_astro_max_freeflying,  # turn off the baked world->base fixed joint, so the base is free-flying
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, max_depenetration_velocity=10.0, enable_gyroscopic_forces=True
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        # The body comes authored FRD, rotors on body -Z. A 180° rotation about X, written as a quat in
        # w, x, y, z order, lands it upright, rotors up, in Newton's Z-up world, matching the convention the
        # standalone runtime uses in nexus physics/builders/usd.py.
        rot=(0.0, 1.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},  # passive rotors, the lumped wrench drives the base; no init spin
    ),
    actuators={"dummy": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
)
"""Astro Max articulation (base body_frd + 4 rotors), spawned from the converted USD."""


@configclass
class AstroMaxGoToEnvCfg(GoToEnvCfg):
    """The GoTo task on the Astro Max: supplies the robot + its FRD frame to the vehicle-agnostic env."""

    def __post_init__(self):
        super().__post_init__()  # the generic GoTo cfg: Newton MJWarp + per-rotor obs/caps
        self.robot = ASTRO_MAX_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        self.base_body = "body_frd"  # Astro Max base link
        self.thrust_sign = -1.0  # FRD: "up" toward the rotors is body -z


gym.register(
    id="Newton-AstroMax-GoTo-Direct-v0",
    entry_point=f"{_TASK}.goto_env:GoToEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_TASK}.config.astro_max:AstroMaxGoToEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_TASK}.agents.rsl_rl_ppo_cfg:GoToPPORunnerCfg",
    },
)
