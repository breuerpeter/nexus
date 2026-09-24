"""Trained-policy flight: fly an exported RL policy through the standalone runtime, recorded to a
Rerun ``.rrd``. This is the **deploy half** of the round-trip: it takes an exported ``policy.pt``
and flies it, no Isaac Lab involved.

The training half lives in the separate ``nexus-rl`` Isaac Lab project, ``--project
nexus-rl``; *this* runs on **standalone Newton** via ``uv``, with no Isaac Lab and no container. It
loads the exported ``policy.pt`` and flies it with the single-body :class:`RigidBodyRotors`, the
same per-rotor model the policy trained against, closing the loop: train on Isaac-Lab-on-Newton →
deploy on the core.

**The shape.** A zero-arg, self-contained script: it assembles its own orchestrator, the
example-owned :mod:`assembly`, and hosts it via ``Sim.from_orchestrator`` + ``sim.operator``. The
controller builds its observation from the ground-truth ``meas.state``, the single train↔deploy obs
source, and runs its TorchScript inference at the host seam; everything else runs CUDA-graph
captured. The in-process operator sequences the waypoints, advancing on arrival, since the policy is
goal-relative, so each arrival hands it a fresh single-goal problem, and owns the run's end.

The policy is the hosted, content-addressed :data:`POLICY_ASSET`, sha-verified into the asset
cache as the vehicle Universal Scene Description (USD) files are; pass ``--policy`` to fly a fresh local
export instead, which is how the RL CI deploys its just-trained checkpoint.

    uv run --extra policy -m nexus.examples goto_policy
    uv run --extra policy -m nexus.examples goto_policy --policy <path-to>/exported/policy.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import nexus as na
from nexus._src.build.launch import resolve_scenario
from nexus._src.config import LaunchConfig
from nexus._src.rendering import rtx_renderer
from nexus.examples._lib import dump_run
from nexus.examples.controllers.policy.assembly import build_policy_orchestrator

# Everything this demo is, in one place: zero required args, the configuration IS the example;
# --policy optionally deploys a fresh local export instead of the hosted checkpoint.
VEHICLE = "astro_max_base"
MAX_STEPS = 6000  # safety cap; the operator ends the run on mission completion
WAYPOINTS = [(1.5, 1.0, 1.5), (-1.5, 1.0, 2.0), (-0.5, -0.5, 1.8)]  # a small tour
# The hosted default policy, content-addressed and sha-verified, trained by nexus-rl on this
# vehicle's USD-authored thrust map; see docs/examples/isaac-lab-rl.md for the training recipe.
POLICY_ASSET = {"name": "goto_policy", "sha256": "6b3edb018f540934bb0aa2c0a23be357684d3a911c9fd986978041a4473902d0"}


def _resolve_policy(override: str | None) -> str:
    """The exported TorchScript policy this flight deploys: the ``--policy`` override, a fresh
    local export, when given, else the hosted :data:`POLICY_ASSET` fetched + sha-verified into the
    asset cache, offline after the first fetch.
    """
    if override:
        if not os.path.isfile(override):
            raise SystemExit(f"--policy {override!r} is not a file (expected an exported policy.pt)")
        return override
    from nexus._src.assets.resolver import CDN, fetch

    name, sha = POLICY_ASSET["name"], POLICY_ASSET["sha256"]
    try:
        return str(fetch(f"{CDN}/policies/{name}-{sha}.pt", sha, filename=f"{name}.pt"))
    except Exception as exc:
        raise SystemExit(
            f"could not fetch the hosted policy ({exc}): pass --policy /path/to/exported/policy.pt "
            "(train + export one: `uv run --project nexus-rl python nexus-rl/scripts/rsl_rl/train.py`)"
        ) from exc


def main() -> None:
    ap = argparse.ArgumentParser(description="Fly the goto waypoint tour with a trained policy.")
    ap.add_argument(
        "--policy", default=None, metavar="PT",
        help="exported TorchScript policy to deploy (default: the hosted, sha-verified checkpoint)",
    )  # fmt: skip
    # parse_known_args: the launcher passes the shared diagnostics flags (--profile/...) through.
    args, _ = ap.parse_known_args()
    policy = _resolve_policy(args.policy)
    na.logger.info(f"[policy] policy={policy}  {len(WAYPOINTS)} waypoints")
    launch = LaunchConfig().set_vehicle(VEHICLE)
    launch.runtime.device = "cuda"  # prefer CUDA; resolve_device falls back to CPU when there is none
    builder, _resolved, cfg = resolve_scenario(launch)
    orch = build_policy_orchestrator(
        cfg,
        policy_path=policy,
        vehicle_builder=builder,
        max_steps=MAX_STEPS,
        rerun=True,  # the .rrd is the demo's artifact
        renderer_factory=rtx_renderer(builder, cfg),  # the Kit peer, when the vehicle authors RTX sensors
    )
    with na.Sim.from_orchestrator(orch) as sim:
        sim.operator.set_mission(WAYPOINTS)  # the operator sequences these, advances on arrival, owns the stop
        t0 = time.time()
        sim.run()  # blocks until the mission completes, or the safety cap
        wall = time.time() - t0

    traj = sim.physics[sim.base_body].history()
    final_pos = np.asarray(traj[-1].position) if traj else np.full(3, np.nan)
    reached = sim.operator.reached
    # Measure to the goal the policy is actively tracking, clamped to the last waypoint on completion,
    # not the last one already passed, so a steps-truncated run reports the live tracking error.
    target = np.asarray(WAYPOINTS[min(reached, len(WAYPOINTS) - 1)])
    final_err = float(np.linalg.norm(final_pos - target))
    control_steps = int(sim.results().get("control_steps", len(traj)))
    throughput = round(control_steps / wall, 1) if wall > 0 else 0.0
    stats = {
        "waypoints": len(WAYPOINTS),
        "reached": reached,
        "control_steps": control_steps,
        "deploy_steps_per_sec": throughput,  # control-loop throughput: sensors→policy→actuator→physics
        "final_tracking_error_m": round(final_err, 4),
    }
    na.logger.info(
        f"[policy] reached {reached}/{len(WAYPOINTS)} waypoints; final pos={np.round(final_pos, 3)} "
        f"err={final_err:.3f} m; {throughput:.0f} steps/s",
    )
    na.logger.info(f"[policy] {json.dumps(stats)}")
    # Evaluation artifacts: flown trajectory + the waypoint mission, with arrival times. The CI
    # harness interpolates the position reference and scores Absolute Pose Error (APE) + the preceding stats.
    dump_run(sim, "goto_policy", stats=stats, waypoints=WAYPOINTS, arrival_times=sim.operator.arrival_times)


if __name__ == "__main__":
    main()
