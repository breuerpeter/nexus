"""Assemble the core Loop around the :class:`TrainedPolicyController`.

This example-owned assembly, moved out of core since PX4 is the one first-class control path, wires
the same core components as the PX4 assembly: ``NewtonPhysics`` on the articulated vehicle from its Universal
Scene Description (USD) file, the rotor chain PX4 flies, the passthrough estimator, and the core
``Loop``. The controller builds its own observation from the estimate, the single
train↔deploy obs source, ``nexus_uav.examples._lib.observation``, and runs its TorchScript
inference in a host stage; everything else captures, the captured-host-exchange strategy.
"""

from __future__ import annotations

from nexus_uav._src.build.assembly import resolve_device
from nexus_uav._src.build.launch import required_sensors
from nexus_uav._src.core import Clock, Loop, logger
from nexus_uav._src.physics import NewtonPhysics
from nexus_uav._src.vehicle.estimators import GroundTruthEstimator

# The RL task's step [s]: the policy acts at 50 Hz, as it trained, and the rate loop and mixer every tick.
POLICY_DT = 0.02


def build_policy_loop(
    cfg: dict,
    *,
    policy_path: str,
    goal_w=(0.0, 0.0, 1.0),
    thrust_to_weight: float = 1.9,
    omega_max: float = 6.0,
    rate_gain: float = 3.0,
    thrust_sign: float = -1.0,
    max_steps: int | None = 2000,
    physics_substeps: int = 1,
    rerun: bool = False,
    viewer: bool = False,
    record_to_rrd: str | None = None,
    debug: bool = False,
    sink=None,
    vehicle_usd=None,
) -> Loop:
    """Assemble the core Loop with a trained-policy Controller in place of PX4.

    The plant is the articulated vehicle the policy trained on: each rotor's ESC sets its Newton
    motor's speed target, physics steps the motors, and the propellers turn each rotor's speed into its force.
    The policy's **Collective Thrust and Body Rates (CTBR)** action reaches the rotors through the
    controller's mixer, an inner rate loop → ``B⁻¹`` → per-rotor commands. The policy runs every
    ``POLICY_DT / dt`` ticks, at its training rate, and the mixer every tick, as in training, so the policy
    flies through ``Loop.run()`` exactly as PX4 does: same fixed-order tick, same live newton.State.

    The defaults are the values the policy trained against: ``thrust_to_weight=1.9``, ``rate_gain=3.0`` and
    ``omega_max`` matching ``_OMEGA_MAX``; ``thrust_sign=-1`` is the Forward Right Down (FRD) USD convention:
    thrust along −body-z, as the 180°-X flip at start makes it point up.

    ``thrust_to_weight`` here is a **declared training constant**, not a plant parameter: it's part
    of the trained network's action interface, what a full-throttle action *means* to the policy,
    so it must match training even though the USD-derived plant value differs, 2.22 for astro-max.
    Changing it requires retraining the policy, not editing this default.
    """
    import numpy as np
    import warp as wp

    from nexus_uav._src.build.assembly import elements
    from nexus_uav.examples._lib import CtbrParams, build_rotor_mixer_from_model
    from nexus_uav.examples.controllers.policy.controller import TrainedPolicyController

    logger.info(f"device: {resolve_device(cfg)}")
    dt = cfg["physics"]["dt"]
    rtf = cfg["physics"].get("rtf", 0)

    if vehicle_usd is None:
        raise ValueError("vehicle_usd is required (resolve it via build.launch.resolve_scenario)")
    physics = NewtonPhysics(vehicle_usd=vehicle_usd, cfg=cfg)
    robot_mass = float(np.sum(physics.model.body_mass.numpy()))
    # Base-body principal inertia diag for the CTBR inner rate loop, τ = I·gain·Δω, read from the
    # model so the loop is inertia-correct, exactly as training reads it, in goto_env.__init__.
    base_inertia = np.asarray(physics.model.body_inertia.numpy())[0].reshape(3, 3)
    inertia_diag = (float(base_inertia[0, 0]), float(base_inertia[1, 1]), float(base_inertia[2, 2]))
    logger.info(f"policy-loop: robot_mass={robot_mass:.4f} kg, inertia_diag={inertia_diag}, goal={tuple(goal_w)}")

    # The airframe mixer, B⁻¹ + each rotor's thrust map, built once from the model rotor geometry + the USD's rotors,
    # hash-pinned and not in cfg: the controller runs the CTBR mixer, rate loop → B⁻¹, and the rotor chain PX4
    # flies, each rotor's ESC and propeller, takes its commands.
    mixer = build_rotor_mixer_from_model(physics.model, vehicle_usd.rotors(), physics.state0.body_q.numpy())
    commands, forces = elements(physics, vehicle_usd)
    # CTBR rate-loop params for the controller's mixer, the values the policy trained with. Inertia read from
    # the model so the loop is inertia-correct, exactly as training reads it. The controller owns the mixer.
    ctbr_params = CtbrParams()
    ctbr_params.thrust_to_weight = float(thrust_to_weight)
    ctbr_params.weight = float(robot_mass) * 9.81
    ctbr_params.thrust_sign = float(thrust_sign)
    ctbr_params.omega_max = float(omega_max)
    ctbr_params.rate_gain = float(rate_gain)
    ctbr_params.inertia = wp.vec3(float(inertia_diag[0]), float(inertia_diag[1]), float(inertia_diag[2]))
    ctbr_params.gyro_ff = 0.0
    controller = TrainedPolicyController(
        policy_path=policy_path,
        goal_w=goal_w,
        device="cpu",
        mixer=mixer,
        ctbr_params=ctbr_params,
        policy_period=max(1, round(POLICY_DT / dt)),
    )
    # The sensors whose class requires something by name, such as the RTX sensors, and what they require, such
    # as the render stage, which the loop hands each at connect and stops with the run.
    sensors, required, started = required_sensors(physics, vehicle_usd, cfg)

    # Optional Rerun recording of the rollout: viewer → serve live on :9876; not viewer → write the
    # full .rrd, to record_to_rrd, else a timestamped default.
    if sink is None and rerun:
        from nexus_uav._src.logging import build_logger

        sink = build_logger(
            physics.model, viewer=viewer, record_to_rrd=record_to_rrd, debug=debug,
            # The viewer's Settings tab: the scenario cfg + this example's effective tunables.
            settings={"scenario": cfg, "example": {
                "policy_path": policy_path, "goal_w": goal_w, "thrust_to_weight": thrust_to_weight,
                "omega_max": omega_max, "rate_gain": rate_gain, "thrust_sign": thrust_sign,
                "max_steps": max_steps, "physics_substeps": physics_substeps,
            }},
        )  # fmt: skip
    return Loop(
        clock=Clock(dt, rtf=rtf),
        physics=physics,
        commands=commands,
        forces=forces,
        sensors=sensors,
        # The controller builds its own obs from the estimate, the single obs source; no observation
        # sensor, no per-tick torch in the graph.
        estimator=GroundTruthEstimator(),
        controller=controller,
        links=required,  # each RTX sensor takes the render stage at connect
        peers=started,
        logger=sink,
        max_steps=max_steps,
        physics_substeps=physics_substeps,
    )
