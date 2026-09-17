"""Assemble the core Orchestrator around the in-process Proportional Integral Derivative (PID)
:class:`PidController`.

This example-owned assembly, moved out of core because PX4 is the one first-class control path, wires
the same core components as the PX4 assembly: ``NewtonPhysics``, the single-body ``Rotors``
actuator, the device-native observation sensor, and the core ``Orchestrator``. The actuator's
aero/thrust map comes from the vehicle's Universal Scene Description (USD) file.
``Sim.from_orchestrator`` hosts the built orchestrator.

This is also the **determinism authority**: a deterministic in-process controller over the
bit-exact Newton CPU physics gives the bit-reproducible CI gate that real PX4, with its
non-deterministic work-queue interleaving, can't. The core determinism/capture tests import
this builder from here, because examples ship in the wheel.
"""

from __future__ import annotations

from nexus._src.core import Clock, ConstantEnvironment, Orchestrator, logger
from nexus._src.physics import NewtonPhysics
from nexus._src.runtimes.assembly import resolve_device


def build_pid_orchestrator(
    cfg: dict,
    *,
    goal_w=(0.0, 0.0, 1.0),
    gains=None,
    thrust_to_weight: float | None = None,  # None -> the plant's true T/W, derived from the USD thrust map
    moment_scale: float = 0.01,
    spawn=(0.0, 0.0, 2.0),  # where free flight starts in the single-body semi_implicit regime
    max_steps: int | None = 2000,
    physics_substeps: int = 1,
    sink=None,
    rerun: bool = False,
    viewer: bool = False,  # serve or file, one only: False records the .rrd, the demo's artifact, as every example does
    record_to_rrd: str | None = None,
    debug: bool = False,
    vehicle_builder=None,
    renderer_factory=None,
) -> Orchestrator:
    """Assemble the core Orchestrator with the in-process :class:`PidController`.

    The control law's gains are the differentiable design parameters tuned offline by the
    design-optimization example. ``cfg["physics"]["solver"] == "semi_implicit"`` selects the
    collapsed single-body free-flight regime, the same differentiable plant the design-opt example
    tunes its gains on, so a diffsim-tuned controller deploys on the plant the optimizer tuned it for.
    That's the only stable use of semi_implicit for a quad: on the articulated multibody its small
    rotor bodies blow up. ``mujoco``, the default, stays the articulated path, the determinism
    authority.
    """
    import numpy as np

    from nexus._src.vehicle.actuators import check_actuator_model_pairing
    from nexus.examples._lib import RigidBodyRotors, build_rotor_mixer_from_model
    from nexus.examples._lib.observation import WarpObservationSensor
    from nexus.examples.controllers.pid.controller import PidController
    from nexus.examples.controllers.pid.law import DEFAULT_GAINS

    logger.info(f"device: {resolve_device(cfg)}")
    dt = cfg["physics"]["dt"]
    rtf = cfg["physics"].get("rtf", 0)

    if vehicle_builder is None:
        raise ValueError("vehicle_builder is required (resolve it via runtimes.launch.resolve_scenario)")
    builder = vehicle_builder
    act = builder.actuator_params()  # aero/thrust map from the vehicle USD, hash-pinned and not in cfg
    single_body = cfg["physics"].get("solver") == "semi_implicit"
    if single_body:
        from nexus.examples._lib import build_rotor_mixer_from_layout
        from nexus.examples._lib.single_body import collapse_to_single_body

        sb = collapse_to_single_body(builder)  # the one single-body seam: collapse plus rotor layout
        physics = NewtonPhysics(
            model=sb.model,
            cfg={"physics": {"dt": dt, "solver": "semi_implicit", "contacts": False,
                             "spawn": {"pos": tuple(spawn), "attitude": "flip"}}},
        )  # fmt: skip
        robot_mass = sb.mass
        # Layout mixer: the collapsed body has no rotor joints for build_rotor_mixer_from_model to read.
        mixer = build_rotor_mixer_from_layout(sb.offsets, sb.dirs, {"ct": act["ct"], "cd": act["cd"], "rpm_max": act["rpm_max"]})  # fmt: skip
    else:
        physics = NewtonPhysics(vehicle_builder=builder, cfg=cfg)  # articulated plant
        robot_mass = float(np.sum(physics.model.body_mass.numpy()))
        # The airframe mixer built from the model rotor geometry + the settled rest pose.
        mixer = build_rotor_mixer_from_model(physics.model, act, physics.state0.body_q.numpy())
    if thrust_to_weight is None:
        # The action scale = the plant's true thrust-to-weight, from the USD-authored thrust map, not a
        # declared constant: action = +1 means the plant's real max thrust. DEFAULT_GAINS' altitude
        # gains use this scale, see law.py.
        thrust_to_weight = mixer.thrust_to_weight(robot_mass)
    logger.info(
        f"pid-orchestrator: robot_mass={robot_mass:.4f} kg, goal={tuple(goal_w)}, "
        f"single_body={single_body}, t2w={thrust_to_weight:.3f}"
    )

    # The PID law emits direct moments [thrust, m_x, m_y, m_z]; the controller's moment mixer, B⁻¹ with no
    # Collective Thrust and Body Rate (CTBR) rate loop, turns them into per-rotor commands, and the
    # single-body Rotors motor model realizes them: the same seam for the articulated model, with
    # force-free rotors, and for the collapsed one, see actuators/coupling.py.
    actuator = RigidBodyRotors(mixer=mixer, dt=dt, thrust_sign=-1.0, motor_tau=act["tau"])
    check_actuator_model_pairing(actuator, physics.model)  # single-body wrench, a no-op guard
    controller = PidController(
        gains=DEFAULT_GAINS if gains is None else gains,
        goal_w=goal_w,
        thrust_to_weight=thrust_to_weight,
        mixer=mixer,
        weight=robot_mass * 9.81,
        moment_scale=moment_scale,
    )
    # Device-native obs sensor: fills meas.observation with a Warp array so the PID's exchange runs
    # on-device with no per-tick host hop -> the in-process loop is fully capturable / tape-able.
    sensors = [WarpObservationSensor(goal_w=goal_w)]
    # The one runtime seam, see the core assembly: an optional renderer plus its host-rate sensors.
    renderer, extra_sensors = renderer_factory(physics, builder, cfg) if renderer_factory else (None, [])
    sensors += extra_sensors
    # Share the sensor's persistent device goal buffer with the controller, so the operator's
    # accept_setpoint(.assign) reaches the captured obs kernel: the capture contract.
    controller.bind_goal_buffer(sensors[0].goal)
    if sink is None and rerun:  # central Rerun recording, §10: the .rrd of a standard PID run
        from nexus._src.logging import build_logger

        sink = build_logger(
            physics.model, viewer=viewer, record_to_rrd=record_to_rrd, debug=debug,
            # The viewer's Settings tab: the scenario cfg + this example's effective tunables.
            settings={"scenario": cfg, "example": {
                "goal_w": goal_w, "gains": DEFAULT_GAINS if gains is None else gains,
                "thrust_to_weight": thrust_to_weight, "moment_scale": moment_scale, "spawn": spawn,
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
