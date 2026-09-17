"""Open-loop dynamics probe for the single-body per-rotor env: a sanity check of the single-body model.

Two checks, both with start-state randomization off, for a clean level start, and no RL:
  A) Hover sweep: hold omega_des=0 and sweep the collective; report altitude drift + tilt + motor Ω.
     Confirms which collective hovers and that the drone stays level, so the attitude is open-loop stable.
  B) Rate tracking: at the hover collective, step omega_des=[r,0,0]; the base body-rate should track it.

Run: uv run --project nexus-rl python nexus-rl/scripts/probe_hover.py
"""

from __future__ import annotations

import gymnasium as gym
import nexus_rl  # noqa: F401  registers the tasks with gymnasium
import torch
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.utils.sim_launcher import compute_kit_requirements, launch_simulation

TASK = "Newton-AstroMax-GoTo-Direct-v0"


def main() -> None:
    n = 8
    task_id = TASK
    cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    cfg.init_randomize = False  # clean level start for the open-loop probe
    cfg.scene.num_envs = n
    cfg.sim.device = "cuda:0"
    la = {"headless": True, "device": "cuda:0", "enable_cameras": False}
    compute_kit_requirements(cfg, la)
    with launch_simulation(cfg, la):
        env = gym.make(task_id, cfg=cfg, render_mode=None)
        e = env.unwrapped
        robot = e._robot
        adim = gym.spaces.flatdim(e.single_action_space)

        print("\n=== A) hover sweep (omega_des=0, vary collective) ===", flush=True)
        for coll in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6):
            env.reset()
            act = torch.zeros(n, adim, device=e.device)
            act[:, 0] = coll
            z0 = robot.data.root_pos_w[:, 2].mean().item()
            for _ in range(150):
                env.step(act)
            z = robot.data.root_pos_w[:, 2].mean().item()
            om = e._omega.abs().mean().item()
            # ≈ +1 when level, body-z down in the Forward Right Down (FRD) frame
            gravz = robot.data.projected_gravity_b[:, 2].mean().item()
            wmag = robot.data.root_ang_vel_w.norm(dim=1).mean().item()
            fin = bool(torch.isfinite(robot.data.root_pos_w).all())
            print(
                f"  coll={coll:+.2f}  dz={z - z0:+.3f}  alt={z:.2f}  meanΩ={om:5.1f}  "
                f"grav_bz={gravz:+.3f}  |ω|={wmag:.3f}  finite={fin}",
                flush=True,
            )

        print("\n=== B) rate tracking (hover collective, step omega_des) ===", flush=True)
        omega_max = float(e._ctbr_params.omega_max)
        for rate_cmd in (0.2, 0.5):
            env.reset()
            act = torch.zeros(n, adim, device=e.device)
            act[:, 0] = 0.2  # ~hover collective for the single-body model
            act[:, 1] = rate_cmd
            target = rate_cmd * omega_max
            for k in range(40):
                env.step(act)
                if k in (3, 10, 25):
                    wb = e._robot.data.root_ang_vel_b[:, 0].mean().item()
                    print(f"  rate_cmd={rate_cmd} (target ωx={target:.2f})  step{k:3d}: ωx_body={wb:+.3f}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
