"""Assemble the core Orchestrator around the :class:`SamplingMPCController`.

This example-owned assembly, moved out of core since PX4 is the one first-class control path, wires
the **catalog vehicle** Universal Scene Description (USD) collapsed to a single rigid body + the resolved
scene USD, its cost-only obstacle shapes, for example the ``slalom`` pillars, into both the real
single-drone model, stepped by the Orchestrator, and the batched rollout model, ``num_rollouts``
differentiable drones the planner back-props through. The controller's per-tick optimization runs in a
host stage; everything else captures, the captured-host-exchange strategy.
"""

from __future__ import annotations

from nexus_sim._src.build.assembly import resolve_device
from nexus_sim._src.core import Clock, Orchestrator, logger
from nexus_sim._src.physics import NewtonPhysics
from nexus_sim._src.vehicle.estimators import GroundTruthEstimator


def build_sampling_mpc_orchestrator(
    cfg: dict,
    *,
    goal_w=(0.0, 0.0, 2.0),
    spawn=(0.0, 0.0, 2.0),
    num_rollouts: int = 16,
    max_steps: int | None = 3400,
    rerun: bool = False,
    viewer: bool = False,
    record_to_rrd: str | None = None,
    debug: bool = False,
    sink=None,
    vehicle_usd=None,
    renderer_factory=None,
) -> Orchestrator:
    """Assemble the core Orchestrator with the sampling + gradient diffsim Model Predictive Control (MPC).

    Needs a scene with obstacles threaded into ``cfg``: ``resolve_scenario`` with
    ``set_scene("slalom")``; the controller discovers the scene's shapes from the collapsed model.
    The guidance sequences the mission and writes each goal to the setpoint the controller reads.
    """
    from nexus_sim.examples._lib import Rotors, build_rotor_mixer_from_layout
    from nexus_sim.examples._lib.single_body import collapse_to_single_body
    from nexus_sim.examples.controllers.sampling_mpc.controller import SamplingMPCController

    logger.info(f"device: {resolve_device(cfg)}")
    dt = cfg["physics"]["dt"]
    if vehicle_usd is None:
        raise ValueError("vehicle_usd is required (resolve it via build.launch.resolve_scenario)")
    if not cfg.get("scene_usd_path"):
        raise ValueError("the sampling MPC needs a scene with obstacles (resolve with scene='slalom')")

    # The one single-body path: collapse the quad-X USD to the real sim, 1 body, + the batched differentiable
    # rollout, num_rollouts bodies. The collapse loads the scene USD exactly as the full build path does;
    # its authored semantics ride with it: the slalom pillars ship ``physics:collisionEnabled = false``, so
    # they're cost-only. The controller discovers the scene's shapes from the model. The collapse reads the
    # rotor layout/params from the articulated USD before it merges the joints.
    real = collapse_to_single_body(vehicle_usd, count=1, requires_grad=False, cfg=cfg)
    batch = collapse_to_single_body(vehicle_usd, count=num_rollouts, requires_grad=True, cfg=cfg)
    real_model, batch_model = real.model, batch.model
    mass, offsets, dirs, m = real.mass, real.offsets, real.dirs, real.act
    logger.info(f"sampling-mpc-orchestrator: mass={mass:.4f} kg, goal={tuple(goal_w)}, num_rollouts={num_rollouts}")

    # Free single-body diff-sim regime: the pre-built collapsed model, + obstacles, gradient-capable
    # SemiImplicit solver, contacts off since avoidance is the controller's job, Forward Right Down (FRD)
    # flip start pose, rotors-up.
    physics = NewtonPhysics(
        model=real_model,
        cfg={"physics": {"dt": dt, "solver": "semi_implicit", "contacts": False,
                         "spawn": {"pos": tuple(spawn), "attitude": "flip"}}},
    )  # fmt: skip
    # Real-sim actuator: the shared single-body Rotors motor model, the same forward-B kernel the
    # planner rolls out, planner ≡ real. The yaw-reaction coefficient is the vehicle USD's authored ``cd``,
    # the single model authority with no proxy override: the real rotor-drag yaw authority is what lets the
    # sampling planner turn the nose to fly the slalom nose-first.
    actuator = Rotors(
        mixer=build_rotor_mixer_from_layout(offsets, dirs, {"ct": m["ct"], "cd": m["cd"], "rpm_max": m["rpm_max"]}),
        dt=dt,
        motor_tau=m["tau"],
        thrust_sign=-1.0,  # FRD-authored USD: thrust along −body-z; rotors start pointing up
    )
    controller = SamplingMPCController(
        batch_model=batch_model, mass=mass, rotor_offsets=offsets, turning_dirs=dirs,
        ct=m["ct"], rpm_max=m["rpm_max"], goal_w=goal_w, dt=dt, num_rollouts=num_rollouts, reaction_k=m["cd"],
        motor_tau=m["tau"],  # plan with the plant's authored motor lag; planner ≡ real
    )  # fmt: skip
    if sink is None and rerun:
        from nexus_sim._src.logging import build_logger

        # The standard Logger logs the collapsed single-body sim, whose model carries the slalom's cost-only
        # pillar capsules: the scene shows the body + pillars. A collapsed model has no rotor
        # joints, so its props can't spin; spin_visual needs an articulated model.
        sink = build_logger(
            physics.model, viewer=viewer, record_to_rrd=record_to_rrd, debug=debug,
            # The viewer's Settings tab: the scenario cfg + this example's effective tunables.
            settings={"scenario": cfg, "example": {
                "goal_w": goal_w, "spawn": spawn, "num_rollouts": num_rollouts, "max_steps": max_steps,
            }},
        )  # fmt: skip
    # Rendering enters here alone, as in the core assembly: an optional renderer + its host-rate sensors.
    renderer, extra_sensors = renderer_factory(physics, vehicle_usd, cfg) if renderer_factory else (None, [])
    return Orchestrator(
        clock=Clock(dt),
        physics=physics,
        actuator=actuator,
        sensors=extra_sensors,
        estimator=GroundTruthEstimator(),
        controller=controller,
        renderer=renderer,
        logger=sink,
        max_steps=max_steps,
        physics_substeps=1,
    )
