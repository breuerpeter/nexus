"""Assemble the core Loop around the :class:`SamplingMPCController`.

This example-owned assembly, moved out of core since PX4 is the one first-class control path, flies the
**catalog vehicle** from its Universal Scene Description (USD) file as every run does, its Newton motors on
the rotor joints and the rotor chain PX4 flies, through the resolved scene USD, whose cost-only obstacle
shapes, for example the ``slalom`` pillars, the planner avoids. The planner keeps a model of its own: the
vehicle collapsed to a single rigid body, ``num_rollouts`` differentiable drones it back-props through,
with a motor lag derived from the vehicle's motors. The controller's per-tick optimization runs in a
host stage; everything else captures, the captured-host-exchange strategy.
"""

from __future__ import annotations

from nexus_uav._src.build.assembly import resolve_device
from nexus_uav._src.core import Clock, Loop, logger
from nexus_uav._src.physics import NewtonPhysics
from nexus_uav._src.rendering import PEER
from nexus_uav._src.vehicle.estimators import GroundTruthEstimator


def build_sampling_mpc_loop(
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
) -> Loop:
    """Assemble the core Loop with the sampling + gradient diffsim Model Predictive Control (MPC).

    Needs a scene with obstacles threaded into ``cfg``: ``resolve_scenario`` with
    ``set_scene("slalom")``; the controller discovers the scene's shapes from its planning model.
    The guidance sequences the mission and writes each goal to the setpoint the controller reads.
    """
    from nexus_uav._src.build.assembly import rotor_chain
    from nexus_uav.examples._lib.motor import motor_time_constant, rotor_spin_inertia
    from nexus_uav.examples._lib.single_body import collapse_to_single_body
    from nexus_uav.examples.controllers.sampling_mpc.controller import SamplingMPCController

    logger.info(f"device: {resolve_device(cfg)}")
    dt = cfg["physics"]["dt"]
    if vehicle_usd is None:
        raise ValueError("vehicle_usd is required (resolve it via build.launch.resolve_scenario)")
    if not cfg.get("scene_usd_path"):
        raise ValueError("the sampling MPC needs a scene with obstacles (resolve with scene='slalom')")

    # The plant: the articulated vehicle every run flies, from free flight at its start point with its rotors at
    # hover speed, through the rotor chain PX4 flies.
    physics_cfg = {**cfg["physics"], "spawn": {"pos": tuple(spawn), "attitude": "native", "prespin": "hover"}}
    physics = NewtonPhysics(vehicle_usd=vehicle_usd, cfg={**cfg, "physics": physics_cfg})
    commands, forces = rotor_chain(physics, vehicle_usd)
    # The planner's own model: the vehicle collapsed to a single body, num_rollouts of them for the batched
    # differentiable rollout. The collapse loads the scene USD exactly as the full build path does; its
    # authored semantics ride with it: the slalom pillars ship ``physics:collisionEnabled = false``, so
    # they're cost-only. The collapse reads the rotor layout/params from the articulated USD before it
    # merges the joints. The planner's motor lag is the time constant of the vehicle's motors.
    batch = collapse_to_single_body(vehicle_usd, count=num_rollouts, requires_grad=True, cfg=cfg)
    mass, offsets, dirs, m = batch.mass, batch.offsets, batch.dirs, batch.act
    tau = motor_time_constant(physics.model, vehicle_usd.rotor_joints(), m)
    inertia = rotor_spin_inertia(physics.model, vehicle_usd.rotor_joints())
    logger.info(
        f"sampling-mpc-loop: mass={mass:.4f} kg, goal={tuple(goal_w)}, num_rollouts={num_rollouts}, "
        f"motor time constant {tau * 1000:.1f} ms"
    )
    controller = SamplingMPCController(
        batch_model=batch.model, mass=mass, rotor_offsets=offsets, turning_dirs=dirs,
        ct=m["ct"], rpm_max=m["rpm_max"], goal_w=goal_w, dt=dt, num_rollouts=num_rollouts, reaction_k=m["cd"],
        motor_tau=tau,  # plan with the time constant of the vehicle's motors
        rotor_inertia=inertia,  # and with the reaction each rotor's speed change pushes on the airframe
    )  # fmt: skip
    if sink is None and rerun:
        from nexus_uav._src.logging import build_logger

        # The standard Logger logs the articulated plant, whose model carries the slalom's cost-only pillar
        # capsules: the scene shows the vehicle, its props turning with its rotor joints, + the pillars.
        sink = build_logger(
            physics.model, viewer=viewer, record_to_rrd=record_to_rrd, debug=debug,
            # The viewer's Settings tab: the scenario cfg + this example's effective tunables.
            settings={"scenario": cfg, "example": {
                "goal_w": goal_w, "spawn": spawn, "num_rollouts": num_rollouts, "max_steps": max_steps,
            }},
        )  # fmt: skip
    # Rendering enters here alone, as in the core assembly: an optional renderer + its host-rate sensors.
    renderer, extra_sensors = renderer_factory(physics, vehicle_usd, cfg) if renderer_factory else (None, [])
    return Loop(
        clock=Clock(dt),
        physics=physics,
        commands=commands,
        forces=forces,
        sensors=extra_sensors,
        estimator=GroundTruthEstimator(),
        controller=controller,
        renderer=renderer,
        links={} if renderer is None else {PEER: renderer},  # each RTX sensor takes the Kit link at connect
        logger=sink,
        max_steps=max_steps,
        physics_substeps=1,
    )
