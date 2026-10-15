"""RL training trains on the plant the core flies: from mid-air with the rotors at rest, the RL task and the
policy example's run answer one constant command alike. Needs Isaac Lab, which only the `nexus-rl` project
installs, and a CUDA device: the RL leg runs it under `nexus-rl`'s environment.
"""

import numpy as np
import pytest

pytest.importorskip("isaaclab")
pytest.importorskip("torch")

import torch

import nexus_uav as nx
from nexus_uav._src.build.launch import resolve_vehicle_usd
from nexus_uav._src.config import LaunchConfig
from nexus_uav.examples.controllers.policy.assembly import build_policy_loop
from nexus_uav.examples.controllers.policy.goto.geofence import GeofenceGuidance
from tests.usd import sensor_vehicle as sv

pytestmark = pytest.mark.gpu

TASK = "Nexus-AstroMax-GoTo-Direct-v0"
# One constant Collective Thrust and Body Rates (CTBR) action: a collective over the middle and a body rate
# about each axis, yaw included.
ACTION = (0.3, 0.2, -0.3, 0.6)
STEPS = 19  # the RL task's 20 ms steps: 0.38 s
TICKS_PER_STEP = 5  # the core's 4 ms ticks in one of them


class _Constant(torch.nn.Module):
    """A policy that answers every observation with `ACTION`."""

    def __init__(self):
        super().__init__()
        self.register_buffer("action", torch.tensor([ACTION], dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.action.expand(obs.shape[0], 4)


def _body_frame(rows) -> np.ndarray:
    """Each row's angular then linear velocity in the base body's frame, from its world-frame twist."""
    out = []
    for row in rows:
        x, y, z, w = (float(v) for v in row["orientation"])
        r = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        out.append(np.r_[r.T @ np.asarray(row["angular_velocity"]), r.T @ np.asarray(row["linear_velocity"])])
    return np.array(out)


def _core(tmp_path) -> np.ndarray:
    """The policy example's run of `ACTION` from 1 m up with the rotors at rest, each 20 ms."""
    path = tmp_path / "constant.pt"
    torch.jit.script(_Constant()).save(str(path))
    cfg = sv.scenario()
    cfg["physics"]["spawn"] = {"pos": (0.0, 0.0, 1.0), "attitude": "native"}
    vb, _ = resolve_vehicle_usd(LaunchConfig().set_vehicle("astro_max_base").set_scene("empty"))
    loop = build_policy_loop(cfg, policy_path=str(path), vehicle_usd=vb, max_steps=STEPS * TICKS_PER_STEP)
    guidance = GeofenceGuidance(bounds=((-50.0, -50.0, -50.0), (50.0, 50.0, 50.0)))
    with nx.Sim._from_loop(loop, guidance=guidance) as sim:
        guidance.set_mission([(0.0, 0.0, 1.0)])
        sim.run()
    rows = sim[f"vehicle/body/{sim.base_body}"].history()
    ticks = np.rint(rows["t"] / cfg["physics"]["dt"]).astype(int)
    return _body_frame(rows[(ticks > 0) & (ticks % TICKS_PER_STEP == 0)][:STEPS])


def _trained() -> np.ndarray:
    """The RL task's answer to `ACTION` from its start with no randomization, after each 20 ms step."""
    import gymnasium as gym
    import nexus_rl  # noqa: F401  registers the task
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab_tasks.utils.sim_launcher import compute_kit_requirements, launch_simulation

    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    cfg.init_randomize = False
    cfg.scene.num_envs = 1
    cfg.sim.device = "cuda:0"
    launcher = {"headless": True, "device": "cuda:0", "enable_cameras": False}
    compute_kit_requirements(cfg, launcher)
    out = []
    with launch_simulation(cfg, launcher):
        env = gym.make(TASK, cfg=cfg, render_mode=None)
        env.reset()
        robot = env.unwrapped.scene["robot"]
        action = torch.tensor([ACTION], device=env.unwrapped.device)
        for _ in range(STEPS):
            env.step(action)
            d = robot.data
            out.append(np.r_[d.root_ang_vel_b.torch[0].cpu().numpy(), d.root_lin_vel_b.torch[0].cpu().numpy()])
        env.close()
    return np.array(out)


def test_rl_training_trains_on_the_plant_the_core_flies(tmp_path):
    """RL training trains on the plant the core flies.

    Given the RL task and the core's articulated vehicle, when both take one constant command from mid-air with
    the rotors at rest, then their body rates and velocities agree to 1e-3 over 0.4 s.
    """
    core = _core(tmp_path)
    trained = _trained()

    assert np.abs(core - trained).max() < 1e-3
