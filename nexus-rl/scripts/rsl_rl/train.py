"""rsl_rl trainer: Proximal Policy Optimization (PPO) on a Newton-backend quadcopter task, kitless,
exporting the policy for the nexus core's FR-7 ``TrainedPolicyController``.

Task-agnostic, the Isaac Lab convention: ``--task <gym-id>`` selects the registered task; ``import nexus_rl``
registers them and the registry entry points supply the env + PPO cfgs. Kitless: Newton physics +
no Kit camera/visualizer → ``launch_simulation()`` never boots a SimulationApp, since ``needs_kit=False``.

Run from the separate ``nexus-rl`` uv project; its own venv has Isaac Lab, the core doesn't:

    uv run --project nexus-rl python nexus-rl/scripts/rsl_rl/train.py --num_envs 2048 --max_iterations 150
"""

from __future__ import annotations

import argparse
import importlib.metadata as _md
import json
import os
import time

import gymnasium as gym
import nexus_rl  # noqa: F401  registers the tasks with gymnasium
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.utils.sim_launcher import compute_kit_requirements, launch_simulation
from rsl_rl.runners import OnPolicyRunner

DEFAULT_TASK = "Newton-AstroMax-GoTo-Direct-v0"


def final_metrics(runner: OnPolicyRunner, log_dir: str) -> dict:
    """The last iteration's learning metrics, the numbers rsl_rl printed and wrote to the run's event
    file: ``success_rate`` (``Metrics/success_rate``) and ``mean_reward`` (``Train/mean_reward``),
    each averaged over the envs. The CI regression check gates them with a margin, since they're
    tight statistics where the trained policy itself is one draw.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    writer = getattr(runner.logger, "writer", None)
    if writer is not None:
        writer.flush()  # the writer flushes every 10 s, so the final iteration can still sit in its queue
    events = EventAccumulator(log_dir)
    events.Reload()
    tags = events.Tags()["scalars"]
    metrics = {}
    for key, tag in (("success_rate", "Metrics/success_rate"), ("mean_reward", "Train/mean_reward")):
        if tag in tags:
            metrics[key] = round(float(events.Scalars(tag)[-1].value), 4)
    return metrics


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--num_envs", type=int, default=2048)
    ap.add_argument("--max_iterations", type=int, default=250)
    ap.add_argument(
        "--seed", type=int, default=42,
        help="pins the random draws, the initial weights, start states and exploration noise, and not the "
        "trained policy: the GPU physics step is not bit-reproducible, so two runs at one seed end as two "
        "policies with the same statistics",
    )  # fmt: skip
    ap.add_argument("--log_dir", default="/work/iclogs/nexus_rl")
    ap.add_argument("--entropy", type=float, default=None, help="override the agent cfg entropy_coef")
    ap.add_argument("--save_interval", type=int, default=10, help="checkpoint cadence (model_{it}.pt)")
    ap.add_argument("--stats_json", default=None, help="write run stats JSON here (CI regression gate)")
    ap.add_argument(
        "--vehicle_usd", default=None,
        help="local vehicle USD to train on (default: the hosted, sha-verified registry asset)",
    )  # fmt: skip
    args, _ = ap.parse_known_args()

    num_envs = args.num_envs
    iters = args.max_iterations
    log_dir = args.log_dir

    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    if args.vehicle_usd:  # load a local Universal Scene Description (USD) file instead of the resolved registry asset
        env_cfg.robot.spawn.usd_path = args.vehicle_usd
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = "cuda:0"
    env_cfg.seed = args.seed

    launcher_args = {"headless": True, "device": "cuda:0", "enable_cameras": False}
    needs_kit, _, _ = compute_kit_requirements(env_cfg, launcher_args)
    rec = {
        "task": args.task,
        "num_envs": num_envs,
        "iters": iters,
        "needs_kit": bool(needs_kit),
        "physics": type(env_cfg.sim.physics).__name__,
        "thrust_to_weight": float(env_cfg.thrust_to_weight),
    }

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.max_iterations = iters
    # entropy_coef defaults to the task cfg: GoToPPORunnerCfg = 0.01, the reliability fix; --entropy
    # overrides for sweeps. See the agent cfg for why it needs a positive value.
    if args.entropy is not None:
        agent_cfg.algorithm.entropy_coef = args.entropy
    # Checkpoint cadence doesn't affect the trained policy; record_demo.py replays these checkpoints, so
    # keep it dense enough to capture the early learning.
    agent_cfg.save_interval = args.save_interval
    agent_cfg.device = "cuda:0"
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _md.version("rsl-rl-lib"))

    torch.cuda.reset_peak_memory_stats()
    with launch_simulation(env_cfg, launcher_args):
        env = gym.make(args.task, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        t0 = time.time()
        runner.learn(num_learning_iterations=iters)
        rec["train_seconds"] = round(time.time() - t0, 2)
        rec["steps_per_sec"] = round(num_envs * agent_cfg.num_steps_per_env * iters / (time.time() - t0), 1)
        rec.update(final_metrics(runner, log_dir))
        export_dir = os.path.join(log_dir, "exported")
        runner.export_policy_to_jit(path=export_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_dir, filename="policy.onnx")
        rec["export"] = {"jit": os.path.join(export_dir, "policy.pt"), "onnx": os.path.join(export_dir, "policy.onnx")}
    print("[nexus-rl]", json.dumps(rec))
    if args.stats_json:  # for the CI regression check, scripts/ci/check_rl_stats.py
        with open(args.stats_json, "w") as f:
            json.dump(rec, f, indent=2)
    return rec


if __name__ == "__main__":
    main()
