"""Proportional Integral Derivative (PID) waypoint tour, the minimal in-process controller example: the
deterministic, differentiable PID, :mod:`law`, flying the collapsed single-body astro-max around a square,
fully CUDA-graph captured.

This controller is the framework's **determinism authority**, deterministic in-process control over
the bit-exact Newton CPU physics, the bit-reproducible CI gate real PX4 can't give, and the
**design-optimization controller**: its gains are the differentiable parameters
``design_opt/gain_tuning.py`` tunes, and this flight deploys that example's proven configuration,
the collapsed single-body plant + stable gains. This flight is the smallest end-to-end demo of
the in-process shape: assemble the orchestrator, the example-owned :mod:`assembly`, host it via
``Sim.from_orchestrator``, command it through ``sim.operator``.

**The shape.** A zero-arg, self-contained script. On CUDA the whole tick captures into one CUDA
graph, controller + actuator + physics + sensors, the captured-inprocess strategy; on CPU it runs
eager and bit-exact.

    uv run -m nexus.examples pid                # flies + asserts + writes the .rrd
    uv run nexus script nexus.examples pid # the same, RTX-rendered under Kit
"""

from __future__ import annotations

import numpy as np

import nexus as na
from nexus._src.config import LaunchConfig
from nexus._src.runtimes.launch import default_renderer_factory, resolve_scenario
from nexus.examples._lib import dump_run
from nexus.examples.controllers.pid.assembly import build_pid_orchestrator

# Everything this demo is, in one place: zero args by design, the configuration IS the example.
VEHICLE = "astro_max_base"
MAX_STEPS = 8000  # safety cap; the operator ends the run on mission completion
# The proven single-body PID configuration: the collapsed semi_implicit plant + the gains
# design_opt/gain_tuning.py's optimizer converges to; its deploy gate verifies they reach a far
# waypoint and hold it, while the stable-but-undamped hand gains ring around a goal instead of settling.
GAINS = [0.483, 0.464, 0.062, 0.099, 9.961, 3.103, 1.521]
MOMENT_SCALE = 0.12
# A square tour at altitude, from the free-flight start point at (0, 0, 2): four corners, back to start.
# Leg length ~3-4 m, the goal scale the gain tuning targets; short hops excite under-damped ringing
# when the operator switches goals at arrival speed.
WAYPOINTS = [(3.0, 0.0, 2.0), (3.0, 3.0, 2.5), (0.0, 3.0, 2.0), (0.0, 0.0, 2.0)]


def main() -> None:
    launch = LaunchConfig().set_vehicle(VEHICLE)
    launch.runtime.device = "cuda"  # prefer CUDA; resolve_device falls back to CPU when there is none
    launch.runtime.solver = "semi_implicit"  # the collapsed single-body plant this PID's tuning targets
    builder, _resolved, cfg = resolve_scenario(launch)
    orch = build_pid_orchestrator(
        cfg,
        vehicle_builder=builder,
        goal_w=WAYPOINTS[0],
        gains=GAINS,
        moment_scale=MOMENT_SCALE,
        max_steps=MAX_STEPS,
        rerun=True,  # the .rrd is the demo's artifact
        renderer_factory=default_renderer_factory(),  # RTX under `nexus script`, headless otherwise
    )
    with na.Sim.from_orchestrator(orch, reached_m=0.3, final_hold_s=2.0) as sim:
        sim.operator.set_mission(WAYPOINTS)  # the operator sequences these, advances on arrival, owns the stop
        sim.run()

    states = sim.physics[sim.base_body].history()
    q = np.array([s.position for s in states])
    reached = sim.operator.reached
    final = float(np.linalg.norm(q[-1] - np.array(WAYPOINTS[-1])))
    stats = {
        "reached": reached,
        "final_dist_m": round(final, 4),
    }
    na.logger.info(
        f"pid flight: {len(q)} steps, reached {reached}/{len(WAYPOINTS)} waypoints, final dist {final:.3f} m"
    )
    # Evaluation artifacts first: a failed run must still leave its trajectory for diagnosis.
    dump_run(sim, "pid", stats=stats, waypoints=WAYPOINTS, arrival_times=sim.operator.arrival_times)
    assert np.isfinite(q).all(), "trajectory diverged"
    assert reached == len(WAYPOINTS), f"did not reach all waypoints (got {reached}/{len(WAYPOINTS)})"
    assert final < 0.35, f"did not settle on the final waypoint (final dist {final:.3f} m)"
    na.logger.info("OK: flew the square, reached every waypoint and settled")


if __name__ == "__main__":
    main()
