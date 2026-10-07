"""Assemble the core Orchestrator around the :class:`AcadosNMPCController`, the acados
Nonlinear Model Predictive Control (NMPC).

This example-owned assembly, moved out of core because PX4 is the one first-class control path,
wires the **catalog vehicle** Universal Scene Description (USD), the full articulated model, and
the core rotor chain. The NMPC is a pure *tracking* controller, and the min-snap/flatness
planner lives with the **guidance**: the flight file hands it to a ``TrackingGuidance``, which plans
the whole-path ``ReferenceTrajectory`` and writes it to the setpoint the NMPC reads. The NMPC's per-tick Sequential Quadratic
Programming (SQP) Real-Time Iteration (RTI) solve runs in a host stage; everything else captures,
the captured-host-exchange strategy. acados needs provisioning first: ``python -m nexus_sim.examples acados_nmpc --provision``.
"""

from __future__ import annotations

from nexus_sim._src.build.assembly import resolve_device, rotor_chain
from nexus_sim._src.core import Clock, Orchestrator, logger
from nexus_sim._src.physics import NewtonPhysics


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
    vehicle_usd=None,
    renderer_factory=None,
) -> Orchestrator:
    """Assemble the core Orchestrator with the real-time NMPC."""
    import newton
    import numpy as np

    from nexus_sim._src.vehicle.rotors import find_rotor_joints
    from nexus_sim._src.vehicle.sensors import StateSensor
    from nexus_sim.examples.controllers.acados_nmpc.controller import AcadosNMPCController

    logger.info(f"device: {resolve_device(cfg)}")
    dt = cfg["physics"]["dt"]
    if vehicle_usd is None:
        raise ValueError("vehicle_usd is required (resolve it via build.launch.resolve_scenario)")
    m = vehicle_usd.actuator_params()  # aero/thrust map from the vehicle USD, hash-pinned and not in cfg

    # Flown sim: the real articulated astro-max USD. Free articulated regime: spawned in free
    # flight, with no ground-settle, at its native Forward Right Down (FRD) orientation with rotors
    # pre-spun to hover. The DC-motor servos then hold Ω; the pre-spin seeds the solver-integrated
    # rotor state.
    physics = NewtonPhysics(
        vehicle_usd=vehicle_usd,
        cfg={"physics": {"dt": dt, "solver": "mujoco", "contacts": True,
                         "spawn": {"pos": tuple(spawn), "attitude": "native", "prespin": "hover"}}},
    )  # the hover pre-spin reads ct from vehicle_usd's USD params, not cfg  # fmt: skip
    # The core rotor chain, the same one PX4 flies: the rotors' command element, the USD-authored
    # newton.actuators rotor motors physics steps, and the propellers' airflow-aware force element from
    # the solver-integrated Ω. Real motor lag, spinning props.
    joints = vehicle_usd.rotor_joints()  # the joint of each rotor the vehicle USD declares
    commands, forces = rotor_chain(physics, vehicle_usd)

    # NMPC rigid-body params, read from the real model in the NMPC's upright frame. Spawned at FRD, the NMPC
    # frame coincides with world, q_nmpc = identity, so the body-frame moment arms = the world-frame rotor
    # offsets at the authored rest pose; the yaw reaction sign matches the aero kernel's drag, ∝ −rotor_z·z.
    _vel, _pos, bodies, base = find_rotor_joints(physics.model, joints)
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
        from nexus_sim._src.logging import build_logger

        sink = build_logger(
            physics.model, viewer=viewer, record_to_rrd=record_to_rrd, debug=debug,
            # The viewer's Settings tab: the scenario cfg + this example's effective tunables.
            settings={"scenario": cfg, "example": {"spawn": spawn, "max_steps": max_steps}},
        )  # fmt: skip
    # Rendering enters here alone, as in the core assembly: an optional renderer plus its host-rate sensors.
    renderer, extra_sensors = renderer_factory(physics, vehicle_usd, cfg) if renderer_factory else (None, [])
    orch = Orchestrator(
        clock=Clock(dt),
        physics=physics,
        commands=commands,
        forces=forces,
        sensors=[StateSensor(), *extra_sensors],
        controller=controller,
        renderer=renderer,
        logger=sink,
        max_steps=max_steps,
        physics_substeps=1,
    )
    return orch
