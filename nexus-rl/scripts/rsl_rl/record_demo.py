"""Record the Astro Max RL training as a Rerun ``.rrd``: a **swarm of the real Astro Max model**
improving over training. It logs the actual Newton geometry through the same ``ViewerRerun`` the
framework's ``Logger`` drives, so you see the real multirotor meshes, not markers, flying to their goals,
and replays progressively trained policies so the swarm goes from crashing to clean convergence.

Two Rerun timelines:
* ``time``: play it to watch the swarm fly. The takes follow one another: early take = chaos, late
  take = all converging;
* ``train_iter``: the hover-success / mean-distance curves rise across the snapshots.

**It doesn't train.** It's a pure post-training step, the ``play.py`` "play a saved policy"
flow: run ``train.py`` first, which checkpoints ``model_{it}.pt`` natively under ``--log_dir``;
then this loads a set of those checkpoints and rolls each out under ``torch.inference_mode()``,
logging the swarm. So the recording reflects exactly the run ``train.py`` produced; nothing is
re-trained or changed.

    # 1) train, which saves checkpoints under --log_dir:
    uv run --project nexus-rl python nexus-rl/scripts/rsl_rl/train.py \
      --num_envs 2048 --max_iterations 300 --log_dir .rl-artifacts/rl
    # 2) record from those checkpoints:
    uv run --project nexus-rl python nexus-rl/scripts/rsl_rl/record_demo.py \
      --log_dir .rl-artifacts/rl --out astromax_rl.rrd
"""

from __future__ import annotations

import argparse
import importlib.metadata as _md
import os
import re

import gymnasium as gym
import nexus_rl  # noqa: F401  registers the tasks with gymnasium
import rerun as rr
import rerun.blueprint as rrb
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.utils.sim_launcher import launch_simulation
from rsl_rl.runners import OnPolicyRunner

from nexus._src.logging import Logger

TASK = "Newton-AstroMax-GoTo-Direct-v0"

# Fractions of the full training run to show in the montage: one early "chaos" frame, then dense
# through the 30–90% band where this task's success actually rises, since entropy + init-state
# randomization push convergence to ~mid-run and sampling early would waste frames on the flat 0 phase,
# ending converged. Each maps to the nearest available checkpoint. Purely a recording cadence.
SNAPSHOT_FRACS = [0.07, 0.3, 0.45, 0.55, 0.67, 0.8, 1.0]
EP_STEPS = 160  # episode frames to animate per snapshot: ~3 s at 50 Hz, enough to converge
DEMO_SEED = 777  # seed each rollout reset so all snapshots share goals/starts, a fair before/after
# The per-rotor env carries the actual per-rotor motor-speed states in env._omega, so the recording spins
# the rotor meshes at the real simulated speed, not a cosmetic constant: props slow on descent, spin up
# to climb. The lumped fallback has no Ω state, so it falls back to a constant cosmetic rate.
SPIN_RATE = 25.0  # rad/s, the cosmetic fallback for the lumped env, which has no per-rotor Ω state


def _blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/", name="Astro Max swarm: play 'time' to watch them learn"),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="/curves/success_rate", name="hover success rate"),
                rrb.TimeSeriesView(origin="/curves/mean_distance", name="mean distance to goal [m]"),
            ),
            column_shares=[3, 2],
        ),
        rrb.TimePanel(state="collapsed"),
        collapse_panels=True,
    )


def _centered_worlds(env_origins: torch.Tensor, n: int) -> torch.Tensor:
    """Env indices forming a centered ~sqrt(n) x sqrt(n) block of the env grid, so the logged swarm
    is a square around the world origin, not the first-n envs, which are an edge strip of the full
    ~sqrt(num_envs) training grid.
    """
    side = round(n**0.5)
    xs, ys = torch.unique(env_origins[:, 0]), torch.unique(env_origins[:, 1])
    cx = xs[(len(xs) - side) // 2 : (len(xs) - side) // 2 + side]
    cy = ys[(len(ys) - side) // 2 : (len(ys) - side) // 2 + side]
    mask = torch.isin(env_origins[:, 0], cx) & torch.isin(env_origins[:, 1], cy)
    return torch.nonzero(mask, as_tuple=False).squeeze(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="/work/iclogs/nexus_rl", help="where train.py saved the checkpoints")
    ap.add_argument("--num_envs", type=int, default=256, help="rollout/viz envs (a policy is env-agnostic)")
    ap.add_argument("--num_agents", type=int, default=64, help="agents shown in the swarm")
    ap.add_argument("--out", default="astromax_rl.rrd", help="output .rrd path")
    ap.add_argument(
        "--vehicle_usd", default=None,
        help="local vehicle USD to spawn (default: the hosted, sha-verified registry asset)",
    )  # fmt: skip
    args, _ = ap.parse_known_args()
    log_dir, num_envs, n_show, out = args.log_dir, args.num_envs, args.num_agents, args.out

    avail = sorted(
        int(m.group(1))
        for f in (os.listdir(log_dir) if os.path.isdir(log_dir) else [])
        if (m := re.fullmatch(r"model_(\d+)\.pt", f))
    )
    if not avail:
        raise SystemExit(f"no model_*.pt checkpoints in {log_dir!r}: run train.py first")
    iters = avail[-1]
    chosen = sorted({min(avail, key=lambda a: abs(a - round(fr * iters))) for fr in SNAPSHOT_FRACS})
    print(f"[record_demo] checkpoints {avail[0]}..{avail[-1]}; replaying {chosen}", flush=True)

    task = TASK
    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    if args.vehicle_usd:  # load a local Universal Scene Description (USD) file instead of the resolved registry asset
        env_cfg.robot.spawn.usd_path = args.vehicle_usd
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = "cuda:0"
    env_cfg.seed = 42
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = "cuda:0"
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _md.version("rsl-rl-lib"))

    with launch_simulation(env_cfg, {"headless": True, "device": "cuda:0", "enable_cameras": False}):
        env = gym.make(task, cfg=env_cfg, render_mode=None)
        wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wenv, agent_cfg.to_dict(), log_dir=log_dir, device="cuda:0")  # no learn(); load only
        u = env.unwrapped
        pm = env.unwrapped.sim.physics_manager  # the NewtonManager class; owns the real Model/State

        # Real geometry of the whole Model; all envs are parallel Newton worlds. Log only a centered
        # square block of envs, since the first-n envs are an edge strip of the full grid, off to one side;
        # set_world_offsets((0,0,0)) keeps them at their true env-spaced positions, no viewer auto-grid.
        vis = _centered_worlds(u._terrain.env_origins, n_show)
        model = pm.get_model()
        # No shape_color gamma correction: the unified host env is Newton 1.3, the isaac-lab uv extra,
        # whose add_usd already stores shape_color in sRGB. ViewerRerun logs it straight through, so the
        # dark body renders correctly, as the deploy path proves. The old linear→sRGB fixup was for the
        # container's Newton 1.2.x; applying it here would double-encode and wash the black body to grey.
        rec = Logger(model, serve=False, record_to_rrd=out, blueprint=_blueprint())
        rec._viewer.set_visible_worlds(vis.tolist())
        rec._viewer.set_world_offsets((0.0, 0.0, 0.0))

        dt = float(u.step_dt)
        t_global = 0.0
        goals = None
        # Per-rotor spin sign from the joint names, …ccw… = + and …cw… = −, broadcast to all envs.
        spin_dirs = (
            torch.tensor([1.0 if "ccw" in n else -1.0 for n in u._robot.joint_names], device=u.device)
            .unsqueeze(0)
            .expand(u.num_envs, -1)
            .contiguous()
        )
        # Per-rotor env: spin the rotor meshes at the actual motor-speed states, env._omega, integrated
        # into a joint angle. Falls back to a cosmetic constant rate if the model has no per-rotor Ω state.
        rotor_angle = torch.zeros(u.num_envs, len(u._robot.joint_names), device=u.device)
        real_spin = hasattr(u, "_omega") and u._omega.shape[1] == rotor_angle.shape[1]
        for it in chosen:
            runner.load(os.path.join(log_dir, f"model_{it}.pt"))  # the play.py "play a saved policy" flow
            policy = runner.get_inference_policy(device="cuda:0")
            with torch.inference_mode():
                torch.manual_seed(DEMO_SEED)
                wenv.reset()
                goals = u._desired_pos_w[vis].clone()
                obs = wenv.get_observations()
                near = torch.zeros(len(vis), device=u.device)
                for _step in range(EP_STEPS):
                    obs, _, _, _ = wenv.step(policy(obs))
                    u._desired_pos_w[vis] = goals  # pin the shown swarm's goals, so they survive a crash reset
                    if real_spin:  # integrate the real per-rotor motor speed, signed cw/ccw, into the angle
                        rotor_angle += u._omega * spin_dirs * dt
                    else:
                        rotor_angle = SPIN_RATE * t_global * spin_dirs  # cosmetic fallback for the lumped env
                    u._robot.write_joint_position_to_sim(rotor_angle)
                    pm.forward()  # refresh body poses via eval_fk before reading the state
                    rec.log_state(pm.get_state(), t_global)  # logs the real swarm geometry
                    t_global += dt
                    pos = u._robot.data.root_pos_w.torch[vis]
                    near += (torch.norm(pos - goals, dim=1) < 0.5).float()
            succ = float((near >= 0.25 * EP_STEPS).float().mean())
            pos = u._robot.data.root_pos_w.torch[vis]
            mean_dist = float(torch.norm(pos - goals, dim=1).mean())
            rr.set_time("train_iter", sequence=it)
            rr.log("curves/success_rate", rr.Scalars(succ))
            rr.log("curves/mean_distance", rr.Scalars(mean_dist))
            print(f"[record_demo] iter={it:4d} success_rate={succ:.3f} mean_dist={mean_dist:.3f}", flush=True)

        rr.log("world/goals", rr.Points3D(goals.cpu().numpy(), colors=(255, 215, 0), radii=0.08), static=True)
        rec.close()

    print(f"[record_demo] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
