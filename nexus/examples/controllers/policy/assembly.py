"""Assemble the core Orchestrator around the :class:`TrainedPolicyController`.

This example-owned assembly, moved out of core since PX4 is the one first-class control path, wires
the same core components as the PX4 assembly: ``NewtonPhysics``, the full articulated Universal Scene
Description (USD), the single-body ``Rotors`` actuator, a ground-truth :class:`StateSensor`, and the core
``Orchestrator``. The controller builds its own observation from ``meas.state``, the single
train↔deploy obs source, ``nexus.examples._lib.observation``, and runs its TorchScript
inference at the host seam; everything else captures, the captured-host-exchange strategy.
"""

from __future__ import annotations

from nexus._src.core import Clock, ConstantEnvironment, Orchestrator, logger
from nexus._src.physics import NewtonPhysics
from nexus._src.runtimes.assembly import resolve_device


def build_policy_orchestrator(
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
    vehicle_builder=None,
    renderer_factory=None,
) -> Orchestrator:
    """Assemble the core Orchestrator with a trained-policy Controller in place of PX4.

    The actuator is the single-body :class:`RigidBodyRotors` in Collective Thrust and Body Rates (CTBR)
    mode, matching the policy's Isaac-Lab **CTBR** action: collective thrust + body-rate setpoints → mixer
    → per-rotor motor-speed states. The policy thus flies through ``Orchestrator.run()`` exactly as PX4
    does: same fixed-order tick, same live newton.State, no orchestrator changes.

    The standalone applies the wrench **once** per step, with no Isaac-Lab substep dilution, so the
    defaults are the *effective* values the policy trained against: ``thrust_to_weight=1.9``, since
    training used 1.9·NUM_SUBSTEPS, diluted back to 1.9, and ``rate_gain=3.0``, training's
    ``_RATE_GAIN``=12 diluted by NUM_SUBSTEPS=4. ``omega_max`` matches ``_OMEGA_MAX`` and
    ``thrust_sign=-1`` is the Forward Right Down (FRD) USD convention: thrust along −body-z, as the 180°-X
    flip at start makes it point up.

    ``thrust_to_weight`` here is a **declared training constant**, not a plant parameter: it's part
    of the trained network's action interface, what a full-throttle action *means* to the policy,
    so it must match training even though the USD-derived plant value differs, 2.22 for astro-max.
    Changing it requires retraining the policy, not editing this default.
    """
    import numpy as np
    import warp as wp

    from nexus._src.vehicle.actuators import check_actuator_model_pairing
    from nexus._src.vehicle.sensors import StateSensor
    from nexus.examples._lib import CtbrParams, RigidBodyRotors, build_rotor_mixer_from_model
    from nexus.examples.controllers.policy.controller import TrainedPolicyController

    logger.info(f"device: {resolve_device(cfg)}")
    dt = cfg["physics"]["dt"]
    rtf = cfg["physics"].get("rtf", 0)

    if vehicle_builder is None:
        raise ValueError("vehicle_builder is required (resolve it via runtimes.launch.resolve_scenario)")
    builder = vehicle_builder
    act = builder.actuator_params()  # aero/thrust map from the vehicle USD; hash-pinned, not in cfg
    physics = NewtonPhysics(vehicle_builder=builder, cfg=cfg)
    robot_mass = float(np.sum(physics.model.body_mass.numpy()))
    # Base-body principal inertia diag for the CTBR inner rate loop, τ = I·gain·Δω, read from the
    # model so the loop is inertia-correct, exactly as training reads it, in goto_env.__init__.
    base_inertia = np.asarray(physics.model.body_inertia.numpy())[0].reshape(3, 3)
    inertia_diag = (float(base_inertia[0, 0]), float(base_inertia[1, 1]), float(base_inertia[2, 2]))
    logger.info(
        f"policy-orchestrator: robot_mass={robot_mass:.4f} kg, inertia_diag={inertia_diag}, goal={tuple(goal_w)}"
    )

    # The airframe mixer, B⁻¹ + forward B + thrust map, built once from the model rotor geometry + the USD
    # thrust map; split across the seam: the controller runs the CTBR mixer, rate loop → B⁻¹, the actuator
    # runs the single-body motor model, per-rotor cmd → lag → forward B → wrench. Both derive from this one
    # RotorMixer so they can't drift.
    mixer = build_rotor_mixer_from_model(physics.model, act, physics.state0.body_q.numpy())
    actuator = RigidBodyRotors(
        mixer=mixer, dt=dt, thrust_sign=thrust_sign, motor_tau=act["tau"]
    )  # τ from the vehicle USD, the single authority
    check_actuator_model_pairing(actuator, physics.model)  # single-body wrench; a no-op guard
    # CTBR rate-loop params for the controller's mixer: the deploy effective values, substeps=1; the policy
    # trained on the ×NUM_SUBSTEPS-baked twins, diluted back here. Inertia read from the model so the loop
    # is inertia-correct, exactly as training reads it. The controller owns the mixer, airframe-aware.
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
    )
    # Ground-truth state through the neutral Measurement seam: the controller builds its own obs
    # from meas.state, the single obs source; no observation sensor, no per-tick torch in the graph.
    sensors = [StateSensor()]
    # The one runtime seam, see the core assembly: an optional renderer + its host-rate sensors.
    renderer, extra_sensors = renderer_factory(physics, builder, cfg) if renderer_factory else (None, [])
    sensors += extra_sensors

    # Optional Rerun recording of the rollout: viewer → serve live on :9876; not viewer → write the
    # full .rrd, to record_to_rrd, else a timestamped default.
    if sink is None and rerun:
        from nexus._src.logging import build_logger

        sink = build_logger(
            physics.model, viewer=viewer, record_to_rrd=record_to_rrd, debug=debug,
            # The viewer's Settings tab: the scenario cfg + this example's effective tunables.
            settings={"scenario": cfg, "example": {
                "policy_path": policy_path, "goal_w": goal_w, "thrust_to_weight": thrust_to_weight,
                "omega_max": omega_max, "rate_gain": rate_gain, "thrust_sign": thrust_sign,
                "max_steps": max_steps, "physics_substeps": physics_substeps,
            }},
        )  # fmt: skip
    return Orchestrator(
        clock=Clock(dt, rtf=rtf),
        environment=ConstantEnvironment(),
        physics=physics,
        actuator=actuator,
        sensors=sensors,
        controller=controller,
        renderer=renderer,
        logger=sink,
        max_steps=max_steps,
        physics_substeps=physics_substeps,
    )
