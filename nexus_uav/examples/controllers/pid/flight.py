"""Proportional Integral Derivative (PID) waypoint tour, the minimal controller example: the
deterministic, differentiable PID, :mod:`law`, flying the articulated astro-max around a square, fully
CUDA-graph captured.

This controller is the framework's **determinism authority**, deterministic control over
the bit-exact Newton CPU physics, the bit-reproducible CI gate real PX4 can't give, and the
**design-optimization controller**: its gains are the differentiable parameters
``design_opt/gain_tuning.py`` tunes on the articulated plant this flight flies, and this flight deploys
the gains it converges to. This flight is the smallest end-to-end demo of
an example's shape: assemble the loop, the example-owned :mod:`assembly`, host it via
``Sim._from_loop``, fly its mission through the guidance it holds.

**The shape.** A zero-arg, self-contained script. On CUDA the whole tick captures into one CUDA
graph, controller + rotor chain + physics + sensors; on CPU it runs
eager and bit-exact.

    uv run -m nexus_uav.examples pid                # flies + asserts + writes the .rrd
"""

from __future__ import annotations

import numpy as np

import nexus_uav as nx
from nexus_uav._src.build.launch import resolve_scenario
from nexus_uav._src.config import LaunchConfig
from nexus_uav._src.guidance import MissionGuidance
from nexus_uav.examples._lib import dump_run
from nexus_uav.examples.controllers.pid.assembly import build_pid_loop

# Everything this demo is, in one place: zero args by design, the configuration IS the example.
VEHICLE = "astro_max_base"
SCENE = "empty"  # flat ground
MAX_STEPS = 8000  # safety cap; the guidance ends the run on mission completion
# The gains design_opt/gain_tuning.py's optimizer converges to on the articulated plant; its deploy gate
# verifies they reach a far
# waypoint and hold it, while the stable-but-undamped hand gains ring around a goal instead of settling.
GAINS = [0.284, 0.291, 0.053, 0.09, 8.529, 2.98, 0.752]
MOMENT_SCALE = 0.12
# A square tour at altitude, from the free-flight start point at (0, 0, 2): four corners, back to start.
# Leg length ~3-4 m, the goal scale the gain tuning targets; short hops excite under-damped ringing
# when the guidance switches goals at arrival speed.
WAYPOINTS = [(3.0, 0.0, 2.0), (3.0, 3.0, 2.5), (0.0, 3.0, 2.0), (0.0, 0.0, 2.0)]


def main() -> None:
    launch = LaunchConfig().set_vehicle(VEHICLE).set_scene(SCENE)
    launch.runtime.device = "cuda"  # prefer CUDA; resolve_device falls back to CPU when there is none
    vehicle_usd, _resolved, cfg = resolve_scenario(launch)
    orch = build_pid_loop(
        cfg,
        vehicle_usd=vehicle_usd,
        goal_w=WAYPOINTS[0],
        gains=GAINS,
        moment_scale=MOMENT_SCALE,
        max_steps=MAX_STEPS,
        rerun=True,  # the .rrd is the demo's artifact
    )
    guidance = MissionGuidance(reached_m=0.3, final_hold_s=2.0)
    with nx.Sim._from_loop(orch, guidance=guidance) as sim:
        guidance.set_mission(WAYPOINTS)  # the guidance sequences these, advances on arrival, ends the run
        sim.run()

    states = sim[f"vehicle/body/{sim.base_body}"].history()
    q = np.asarray(states["position"], dtype=float)
    reached = guidance.reached
    final = float(np.linalg.norm(q[-1] - np.array(WAYPOINTS[-1])))
    stats = {
        "reached": reached,
        "final_dist_m": round(final, 4),
    }
    nx.logger.info(
        f"pid flight: {len(q)} steps, reached {reached}/{len(WAYPOINTS)} waypoints, final dist {final:.3f} m"
    )
    # Evaluation artifacts first: a failed run must still leave its trajectory for diagnosis.
    dump_run(sim, "pid", stats=stats, waypoints=WAYPOINTS, arrival_times=guidance.arrival_times)
    assert np.isfinite(q).all(), "trajectory diverged"
    assert reached == len(WAYPOINTS), f"did not reach all waypoints (got {reached}/{len(WAYPOINTS)})"
    assert final < 0.35, f"did not settle on the final waypoint (final dist {final:.3f} m)"
    nx.logger.info("OK: flew the square, reached every waypoint and settled")


if __name__ == "__main__":
    main()
