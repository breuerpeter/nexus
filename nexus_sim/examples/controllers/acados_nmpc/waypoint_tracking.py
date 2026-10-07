"""acados Nonlinear Model Predictive Control (NMPC): fly the astro-max through a curving seven-waypoint
course with a real-time nonlinear Model Predictive Control (MPC). It solves the optimal-control problem
directly with **acados**, the High Performance Interior Point Method (HPIPM) solver and Sequential
Quadratic Programming (SQP) real-time iteration, to track a smooth **min-snap polynomial** reference
through the waypoints, flown nose-first. ``obstacle_slalom.py`` is its
counterpart: a differentiable-simulation sampling MPC for the obstacle-avoidance regime an analytic NMPC
can't easily handle.

**The shape.** A zero-arg, self-contained script: it assembles its own orchestrator, the example-owned
:mod:`assembly`, and hosts it via ``Sim.from_orchestrator`` + ``sim.guidance``. The min-snap +
differential-flatness planner lives with the **guidance**: ``sim.guidance.set_mission(WAYPOINTS)``
plans the whole-path flat-state reference and writes it to the setpoint the NMPC reads, as a
``ReferenceTrajectory``; the controller just *tracks* it. The flat-state reference, position + attitude +
body-rate + thrust feedforward, is the quadrotor differential-flatness map with a velocity-aligned,
nose-first yaw: N=20 shooting nodes, HPIPM partial
condensing, SQP Real Time Iteration (RTI) re-solved every control tick. The ruckig ``FlatnessReference``
remains as the jerk-limited fallback planner.

acados is **not** a plain pip dependency: it's a C library this machine compiles, and it generates and
compiles a C solver. Provision it once, then fly:

    uv run --extra acados -m nexus_sim.examples acados_nmpc --provision   # fetches + builds acados
    uv run --extra acados -m nexus_sim.examples acados_nmpc    # flies + asserts + writes the .rrd

From an installed package the same two commands are ``python -m nexus_sim.examples acados_nmpc``, with and
without ``--provision``, after ``pip install 'nexus-sim[acados]'``. See :mod:`provision` for what the
provisioning builds and where.

Needs a CUDA device, for the Newton sim, and CMake and a C compiler, for acados and its codegen. The first
flight compiles the generated solver, taking a few seconds; later flights reuse it.
"""

from __future__ import annotations

import sys

# First, before the example builds anything: --provision builds acados and ends, and a machine that lacks the
# acados extra or the acados build gets one message that names what to run. On a machine that has both,
# require() makes the acados tree usable in this process, overridable via ACADOS_SOURCE_DIR.
from nexus_sim.examples.controllers.acados_nmpc.provision import provision, require

if "--provision" in sys.argv[1:]:
    provision()
    sys.exit(0)
try:
    require()
except RuntimeError as missing:
    sys.exit(str(missing))

import numpy as np

import nexus_sim as nx
from nexus_sim._src.build.launch import resolve_scenario
from nexus_sim._src.config import LaunchConfig
from nexus_sim._src.guidance import TrackingGuidance
from nexus_sim._src.rendering import rtx_renderer
from nexus_sim.examples._lib import dump_run
from nexus_sim.examples._lib.min_snap import MinSnapReference
from nexus_sim.examples.controllers.acados_nmpc.assembly import build_acados_orchestrator

# Everything this demo is, in one place: zero args by design, the configuration IS the example.
VEHICLE = "astro_max_base"
SCENE = "empty"  # flat ground
MAX_STEPS = 4500  # safety cap; the guidance ends the run after the reference duration, ~12.6 s, + hold

# A curving course, not straight segments: the waypoints trace a climbing left-hand arc that hooks back
# past the start. The horizontal velocity direction sweeps ~250° smoothly over the ~13 s flight, ≲1.1 rad/s
# yaw rate, well inside the planner's feed-forward clip. The min-snap polynomial rounds it into one
# continuous curve, flown nose-first. The heavy astro-max has limited yaw authority, from small rotor-drag
# coupling, so the sweep must stay gradual, with no hairpin reversal, for nose-first + tight tracking to
# coexist. Fly through wp1..wp6; settle on the goal.
WAYPOINTS = [
    (2.0, 0.5, 3.0),  # climb out, heading ~east
    (4.0, 2.0, 3.5),  # begin the left-hand arc
    (4.5, 4.5, 4.0),  # apex: heading north, top of the climb
    (3.0, 6.5, 4.0),  # arcing west
    (0.5, 7.0, 3.5),  # far side, descending
    (-1.5, 5.5, 3.0),  # hooking south past the start
    (-2.0, 3.0, 2.5),  # goal: settle
]


def main() -> None:
    launch = LaunchConfig().set_vehicle(VEHICLE).set_scene(SCENE)
    launch.runtime.device = "cuda"  # prefer CUDA; resolve_device falls back to CPU when there is none
    builder, _resolved, cfg = resolve_scenario(launch)
    orch = build_acados_orchestrator(
        cfg,
        vehicle_builder=builder,
        max_steps=MAX_STEPS,
        rerun=True,  # the .rrd is the demo's artifact
        renderer_factory=rtx_renderer(builder, cfg),  # the Kit peer, when the vehicle authors RTX sensors
    )
    # The guidance owns the planner. MinSnapReference, the polynomial planner, is a smooth min-snap path
    # plus a smooth yaw polynomial, so nose-first flight tracks cleanly; swap in the ruckig
    # FlatnessReference for the jerk-limited fallback.
    mass = float(orch.physics.model.body_mass.numpy().sum())
    controller = orch.controller  # the NMPC its assembly built, whose telemetry the flight reads after the run
    guidance = TrackingGuidance(planner=lambda waypoints: MinSnapReference(waypoints, mass=mass), final_hold_s=3.0)
    with nx.Sim.from_orchestrator(orch, guidance=guidance) as sim:
        sim.guidance.set_mission(WAYPOINTS)  # the guidance plans the min-snap reference; the NMPC tracks it
        sim.run()

    states = sim.physics[sim.base_body].history()
    traj = np.array([s.position for s in states])
    quats = np.array([s.quat_xyzw for s in states])  # xyzw order
    qx, qy, qz, qw = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    # thrust/up axis · world +z: −body z, since the vehicle's Universal Scene Description (USD) has
    # Forward Right Down (FRD) authoring
    up_z = -(1.0 - 2.0 * (qx * qx + qy * qy))
    tilt = np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0)))
    final = float(np.linalg.norm(traj[-1] - np.array(WAYPOINTS[-1])))
    max_tilt = float(tilt.max())
    # Nose-first: while cruising, the body +x axis should track the horizontal velocity direction, the
    # guidance's velocity-aligned yaw reference. Body +x in world (x,y) compared to the horizontal velocity,
    # over the moving steps; the check skips near-hover steps, where nothing constrains the heading.
    vel = np.array([s.velocity for s in states])  # world linear velocity
    bx = np.stack([1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz)], axis=1)  # body +x, world (x, y)
    bx /= np.linalg.norm(bx, axis=1, keepdims=True) + 1e-9
    vh = vel[:, :2]
    sp = np.linalg.norm(vh, axis=1)
    moving = sp > 0.8  # m/s; ignore near-hover
    cos_a = np.clip((bx[moving, 0] * vh[moving, 0] + bx[moving, 1] * vh[moving, 1]) / (sp[moving] + 1e-9), -1.0, 1.0)
    nose_err = float(np.median(np.degrees(np.arccos(cos_a)))) if moving.any() else 0.0
    # Tracking error compared to the planned reference: evo-style position Absolute Pose Error (APE), world
    # frame, time-synced, with no alignment needed since both are in the sim world frame. The metric that
    # catches a flight-quality regression the end-state `final dist` misses.
    track = np.array(controller.track_err)
    track_rmse = float(np.sqrt(np.mean(track**2))) if track.size else 0.0
    track_max = float(track.max()) if track.size else 0.0
    nx.logger.info(
        f"acados NMPC flight: {len(traj)} steps, final dist {final:.3f} m, pos {np.round(traj[-1], 2)}, "
        f"max tilt {max_tilt:.1f}°, nose-vs-velocity median {nose_err:.1f}°, "
        f"track APE rmse {track_rmse:.3f} m / max {track_max:.3f} m",
    )
    stats = {
        "final_dist_m": round(final, 4),
        "track_rmse_m": round(track_rmse, 4),
        "track_max_m": round(track_max, 4),
        "max_tilt_deg": round(max_tilt, 1),
        "nose_err_deg": round(nose_err, 1),
    }
    # Evaluation artifacts first, before any gate can raise, since a failed run must still leave its
    # trajectory for diagnosis: flown trajectory + the planned flat-state reference, pos + attitude,
    # time-aligned via the guidance's reference anchor. The CI harness scores pose APE, trans + rot.
    dump_run(
        sim,
        "acados_nmpc",
        stats=stats,
        reference=controller.reference,
        reference_t0=guidance.reference_started_at,
    )
    assert np.isfinite(traj).all(), "trajectory diverged"
    assert final < 0.2, f"did not reach the final waypoint (final dist {final:.3f} m)"
    assert nose_err < 25.0, f"not flying nose-first (nose-vs-velocity median {nose_err:.1f}°)"
    assert max_tilt < 45.0, f"flight too aggressive (max tilt {max_tilt:.1f}°)"
    # Tracking gate, the one a regression slips past `final dist`: on the core rotor chain,
    # real DC-motor servos, the near-instant thrust the NMPC's Optimal Control Problem (OCP) assumes, the
    # validated flight tracks at ~0.033 m rmse / 0.055 m max; the old lumped first-order lag bloated it to
    # ~0.07/0.17.
    assert track_rmse < 0.05, f"poor reference tracking (APE rmse {track_rmse:.3f} m)"
    assert track_max < 0.1, f"reference tracking spikes (APE max {track_max:.3f} m)"
    nx.logger.info("OK: reached the waypoint, flew nose-first, gentle flight, tracked the reference")


if __name__ == "__main__":
    main()
