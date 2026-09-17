"""Assemble the core Orchestrator around the :class:`AcadosNMPCController`, the acados
Nonlinear Model Predictive Control (NMPC).

This example-owned assembly, moved out of core because PX4 is the one first-class control path,
wires the **registry vehicle** Universal Scene Description (USD), the full articulated model, and
the unified per-rotor actuator. The NMPC is a pure *tracking* controller, and the min-snap/flatness
planner lives with the **operator**: this attaches the planner factory to the orchestrator as
``reference_planner`` so ``Sim`` hands it to the ``InProcessOperator``, which plans the whole-path
``ReferenceTrajectory`` and feeds ``accept_setpoint``. The NMPC's per-tick Sequential Quadratic
Programming (SQP) Real-Time Iteration (RTI) solve runs at the host seam; everything else captures,
the captured-host-exchange strategy. acados needs provisioning first: ``scripts/setup_acados.sh``.
"""

from __future__ import annotations

from nexus._src.core import Clock, ConstantEnvironment, Orchestrator, logger
from nexus._src.physics import NewtonPhysics
from nexus._src.runtimes.assembly import resolve_device


def build_acados_orchestrator(
    cfg: dict,
    *,
    spawn=(0.0, 0.0, 2.0),
    max_steps: int | None = 1700,
    rerun: bool = False,
    viewer: bool = False,
    record_to_rrd: str | None = None,
    debug: bool = False,
    sink=None,
    vehicle_builder=None,
    renderer_factory=None,
) -> Orchestrator:
    """Assemble the core Orchestrator with the in-process real-time NMPC."""
    import newton
    import numpy as np

    from nexus._src.vehicle.actuators import ArticulatedRotors, check_actuator_model_pairing, find_rotor_joints
    from nexus._src.vehicle.sensors import StateSensor
    from nexus.examples._lib.min_snap import MinSnapReference
    from nexus.examples._lib.reference import FlatnessReference  # noqa: F401 the ruckig fallback planner
    from nexus.examples.controllers.acados_nmpc.controller import AcadosNMPCController

    logger.info(f"device: {resolve_device(cfg)}")
    dt = cfg["physics"]["dt"]
    if vehicle_builder is None:
        raise ValueError("vehicle_builder is required (resolve it via runtimes.launch.resolve_scenario)")
    builder = vehicle_builder
    m = builder.actuator_params()  # aero/thrust map from the vehicle USD, hash-pinned and not in cfg

    # Flown sim: the real articulated astro-max USD. Free articulated regime: spawned in free
    # flight, with no ground-settle, at its native Forward Right Down (FRD) orientation with rotors
    # pre-spun to hover. The DC-motor servos then hold Ω; the pre-spin seeds the solver-integrated
    # rotor state.
    physics = NewtonPhysics(
        vehicle_builder=builder,
        cfg={"physics": {"dt": dt, "solver": "mujoco", "contacts": True,
                         "spawn": {"pos": tuple(spawn), "attitude": "native", "prespin": "hover"}}},
    )  # the hover pre-spin reads ct from vehicle_builder's USD params, not cfg  # fmt: skip
    # The core articulated actuator, the same one PX4 flies: USD-authored newton.actuators rotor
    # motors plus the framework's airflow-aware aero from the solver-integrated Ω. Real motor lag,
    # spinning props.
    actuator = ArticulatedRotors(
        model=physics.model, control=physics.control,
        ct=m["ct"], cd=m["cd"], rpm_max=m["rpm_max"], dt=dt,
        aero_h=m.get("aero_h", 0.0), aero_hforce=m.get("aero_hforce", 0.0),
    )  # fmt: skip
    check_actuator_model_pairing(actuator, physics.model)  # requires the USD-authored rotor motors

    # NMPC rigid-body params, read from the real model in the NMPC's upright frame. Spawned at FRD, the NMPC
    # frame coincides with world, q_nmpc = identity, so the body-frame moment arms = the world-frame rotor
    # offsets at the authored rest pose; the yaw reaction sign matches the aero kernel's drag, ∝ −rotor_z·z.
    _vel, _pos, bodies, base = find_rotor_joints(physics.model)
    rest = physics.model.state()
    newton.eval_fk(physics.model, physics.model.joint_q, physics.model.joint_qd, rest)
    bq = rest.body_q.numpy()
    mass = float(physics.model.body_mass.numpy().sum())
    base_inertia = np.asarray(physics.model.body_inertia.numpy()[base]).reshape(3, 3)
    rotor_offsets = (bq[bodies, :3] - bq[base, :3]).astype(np.float32)  # (n,3) body-frame moment arms
    rotor_zz = 1.0 - 2.0 * (bq[bodies, 3] ** 2 + bq[bodies, 4] ** 2)  # rotor body +z, world z-component
    turning_dirs = (-np.sign(rotor_zz)).astype(np.float32)  # clockwise/counter-clockwise yaw-reaction sign
    logger.info(f"acados-orchestrator: mass={mass:.4f} kg")
    controller = AcadosNMPCController(
        mass=mass, inertia=base_inertia, rotor_offsets=rotor_offsets, turning_dirs=turning_dirs,
        ct=m["ct"], rpm_max=m["rpm_max"], dt=dt,
    )  # fmt: skip

    if sink is None and rerun:
        from nexus._src.logging import build_logger

        sink = build_logger(
            physics.model, viewer=viewer, record_to_rrd=record_to_rrd, debug=debug,
            # The viewer's Settings tab: the scenario cfg + this example's effective tunables.
            settings={"scenario": cfg, "example": {"spawn": spawn, "max_steps": max_steps}},
        )  # fmt: skip
    # The one runtime seam, see the core assembly: an optional renderer plus its host-rate sensors.
    renderer, extra_sensors = renderer_factory(physics, builder, cfg) if renderer_factory else (None, [])
    orch = Orchestrator(
        clock=Clock(dt),
        environment=ConstantEnvironment(),
        physics=physics,
        actuator=actuator,
        sensors=[StateSensor(), *extra_sensors],
        controller=controller,
        renderer=renderer,
        logger=sink,
        max_steps=max_steps,
        physics_substeps=1,
    )
    # The operator owns the planner: expose a factory the Sim hands to the
    # InProcessOperator, which plans the whole-path reference and feeds accept_setpoint(ReferenceTrajectory).
    # MinSnapReference, the polynomial planner, is a smooth min-snap path plus a smooth yaw
    # polynomial, so nose-first flight tracks cleanly; swap in the ruckig FlatnessReference for the
    # jerk-limited fallback.
    orch.reference_planner = lambda waypoints: MinSnapReference(waypoints, mass=mass)
    return orch
