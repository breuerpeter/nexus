"""Sampling Model Predictive Control (MPC): fly the astro-max to a target THROUGH an obstacle field
with a differentiable-simulation, sampling + gradient model-predictive controller. The framework
port of NVIDIA's ``example_diffsim_drone``, and the counterpart to ``waypoint_tracking.py``:

  * ``waypoint_tracking.py``: an analytic acados Nonlinear Model Predictive Control (NMPC),
    unbeatable on smooth, free-space tracking.
  * ``obstacle_slalom.py``, this file: the regime where differentiable simulation earns its keep.
    The cost is non-smooth and geometry-driven, obstacle avoidance via signed-distance fields, and
    the landscape is non-convex, go left *or* right around the pillar, so a single analytic solve
    gets trapped in it.

**The shape.** A zero-arg, self-contained script: it assembles its own orchestrator, the
example-owned :mod:`assembly` that wraps controller + actuator + physics around the core components,
and hosts it via ``Sim.from_orchestrator`` + ``sim.operator``. The vehicle is the registry
``astro-max`` and the obstacle pillars ride in the registry ``slalom`` scene, a
Universal Scene Description (USD) file of cost-only capsules. The operator sequences the waypoints:
``set_mission`` → advance on arrival → ``accept_setpoint``; the controller logs its MPC horizon to
``/controller``; the central recorder, always on, logs the physics trajectory + pillars, and the
.rrd is the demo's artifact.

    uv run --extra examples -m nexus.examples sampling_mpc     # flies + asserts + writes the .rrd
    uv run nexus script nexus.examples sampling_mpc      # the same, RTX-rendered under Kit

Needs a Compute Unified Device Architecture (CUDA) device for the batched rollout + graph-captured
optimization.
"""

from __future__ import annotations

import itertools

import numpy as np

import nexus as na
from nexus._src.config import LaunchConfig
from nexus._src.runtimes.launch import default_renderer_factory, resolve_scenario
from nexus.examples._lib import dump_run
from nexus.examples.controllers.sampling_mpc.assembly import build_sampling_mpc_orchestrator

# Everything this demo is, in one place: zero args by design, the configuration IS the example.
VEHICLE = "astro_max_base"
SCENE = "slalom"  # the registry obstacle scene, cost-only pillar capsules
MAX_STEPS = 4000  # safety cap; the operator ends the run on mission completion
# A slalom: three waypoints staggered into a zigzag, not a straight row, each leg a gentle hop. A pillar
# sits mid-way along every leg in the registry 'slalom' scene, so the drone must detour around it.
SPAWN = (0.0, 0.0, 2.0)
WAYPOINTS = [(2.5, 1.2, 2.0), (5.0, -1.0, 2.0), (7.5, 0.7, 2.0)]
PILLAR_RADIUS = 0.4  # must match the slalom scene's authored pillars, see scripts/assets/author_slalom_scene.py


def main() -> None:
    launch = LaunchConfig().set_vehicle(VEHICLE).set_scene(SCENE)
    launch.runtime.device = "cuda"  # prefer CUDA; resolve_device falls back to CPU when there is none
    builder, _resolved, cfg = resolve_scenario(launch)
    orch = build_sampling_mpc_orchestrator(
        cfg,
        vehicle_builder=builder,
        max_steps=MAX_STEPS,
        rerun=True,  # the .rrd is the demo's artifact
        renderer_factory=default_renderer_factory(),  # RTX under `nexus script`, headless otherwise
    )
    # reached_m=0.5 matches the demo's leg spacing: the operator advances to the next waypoint this close;
    # final_hold_s lets the stochastic MPC settle on the final waypoint before the operator ends the run.
    with na.Sim.from_orchestrator(orch, reached_m=0.5, final_hold_s=4.0) as sim:
        sim.operator.set_mission(WAYPOINTS)  # the operator sequences these; the planner avoids the pillars
        sim.run()

    traj = sim.physics[sim.base_body].history()
    q = np.array([s.position for s in traj])
    quats = np.array([s.quat_xyzw for s in traj])  # xyzw order
    reached = sim.operator.reached

    # Clearance to the nearest pillar axis in xy → must exceed the pillar radius, so it cleared every pillar.
    # Pillars sit at the mid-point of each mission leg, the same geometry the scene USD authors.
    legs = [SPAWN, *WAYPOINTS]
    pillars_xy = np.array([(0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])) for a, b in itertools.pairwise(legs)])
    clearance = np.min(np.linalg.norm(pillars_xy[None, :, :] - q[:, None, :2], axis=2), axis=1)
    # Tilt from level: the thrust/up axis is −body-z, as the Forward Right Down (FRD) authored USD has it;
    # its world +z component.
    qx, qy, qz, qw = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    up_z = -(1.0 - 2.0 * (qx * qx + qy * qy))
    tilt = np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0)))
    # Nose-first: while cruising, the body +x axis should track the horizontal velocity direction, the
    # velocity-aligned heading cost. Body +x in world xy compared to the horizontal velocity, over the moving steps.
    vel = np.array([s.velocity for s in traj])  # world linear velocity
    bx = np.stack([1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz)], axis=1)  # body +x, world xy
    bx /= np.linalg.norm(bx, axis=1, keepdims=True) + 1e-9
    sp = np.linalg.norm(vel[:, :2], axis=1)
    moving = sp > 0.8  # m/s; ignore near-hover, where nothing constrains the heading
    cos_a = np.clip((bx[moving, 0] * vel[moving, 0] + bx[moving, 1] * vel[moving, 1]) / (sp[moving] + 1e-9), -1.0, 1.0)
    nose_err = float(np.median(np.degrees(np.arccos(cos_a)))) if moving.any() else 0.0

    final = float(np.linalg.norm(q[-1] - np.array(WAYPOINTS[-1])))
    min_clear = float(clearance.min())
    max_tilt = float(tilt.max())
    cruise_z = WAYPOINTS[0][2]  # every waypoint is at this altitude
    alt_dev = float(np.abs(q[:, 2] - cruise_z).max())
    # Path wiggle: horizontal path length / the direct start→wp1→wp2→goal length. 1.0 is a straight shot;
    # an oscillatory flight inflates it. Catches the "wanders / doesn't converge" regression that the
    # end-state `final dist` misses: the drone can wobble the whole way yet still end near the goal.
    legs = np.array([q[0, :2], *[w[:2] for w in WAYPOINTS]])
    direct = float(np.sum(np.linalg.norm(np.diff(legs, axis=0), axis=1)))
    path_len = float(np.sum(np.linalg.norm(np.diff(q[:, :2], axis=0), axis=1)))
    wiggle = path_len / direct
    na.logger.info(
        f"sampling MPC flight: {len(q)} steps, reached {reached}/{len(WAYPOINTS)} waypoints, "
        f"final dist {final:.3f} m, pos {np.round(q[-1], 2)}, "
        f"min pillar clearance {min_clear:.2f} m (pillar radius {PILLAR_RADIUS:.2f} m), "
        f"max tilt {max_tilt:.1f}°, nose-vs-velocity median {nose_err:.1f}°, "
        f"max altitude deviation {alt_dev:.2f} m, path wiggle {wiggle:.2f}×",
    )
    stats = {
        "reached": reached,
        "final_dist_m": round(final, 4),
        "min_clearance_m": round(min_clear, 3),
        "max_tilt_deg": round(max_tilt, 1),
        "path_wiggle": round(wiggle, 3),
        "nose_err_deg": round(nose_err, 1),
    }
    # Evaluation artifacts first, before any gate can raise, so a failed run still leaves its
    # trajectory for diagnosis. Flown trajectory + the waypoint mission with arrival times; the CI
    # harness interpolates the position reference. NB the flight legitimately detours around the
    # pillars, so the harness monitors the position Absolute Pose Error (APE) rather than gating on
    # it; the gates below are the behavior authority.
    dump_run(sim, "sampling_mpc", stats=stats, waypoints=WAYPOINTS, arrival_times=sim.operator.arrival_times)
    assert np.isfinite(q).all(), "trajectory diverged"
    assert reached == len(WAYPOINTS), f"did not reach all waypoints (got {reached}/{len(WAYPOINTS)})"
    # Convergence: tightened from 0.5. The well-tuned flight settles to ~0.22 m; an under-damped one that
    # wanders the target the whole hold ends ~0.4 m off, the regression this gate now catches.
    assert final < 0.35, f"did not settle on the final waypoint (final dist {final:.3f} m)"
    assert min_clear > PILLAR_RADIUS, (
        f"hit a pillar (min clearance {min_clear:.2f} m < pillar radius {PILLAR_RADIUS:.2f} m)"
    )
    assert max_tilt < 60.0, f"flight too aggressive (max tilt {max_tilt:.1f}°)"
    # The run records nose-first but doesn't gate on it: the winner-take-all sampling planner's
    # heading behavior is realization-sensitive. Measured on the eager path across Random Number
    # Generator (RNG) seeds {0, 1, 7, 1234}, the nose median lands at {13.8°, 25.3°, 96.8°, 74.0°};
    # the heading term is secondary by design, see the controller's cost docstring. A robust
    # nose-first sampling planner is a controller follow-up; the acados example keeps its hard
    # nose-first gate, because the flat reference enforces it robustly.
    assert alt_dev < 0.4, f"altitude wandered (max deviation {alt_dev:.2f} m from cruise height)"
    # Smoothness: a crisp slalom holds ~1.86×; an oscillatory one inflates the path past 2×, the gate the
    # old final-dist-only check missed, because a wandering flight can still end near the goal.
    assert wiggle < 1.95, f"oscillatory / wandering path (wiggle {wiggle:.2f}× the direct route)"
    na.logger.info("OK: flew the slalom, reached every waypoint, cleared every pillar, stayed upright, held altitude")  # fmt: skip


if __name__ == "__main__":
    main()
