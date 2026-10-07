"""The core orchestrator assembly, shared by every run, per §12.

Controller-agnostic by construction: the assembly wires the plant, ``NewtonPhysics`` from nexus_sim's
own ModelBuilder + solvers, the shipped rotor chain, the rotors' command stage and the propellers' force
element around the Newton motors physics steps, the sensor suite authored in
Universal Scene Description (USD) with the site's ambient values, and the Rerun sink around a
**caller-supplied controller**. Which controller flies is a decision one layer up: the launch glue
builds the controller the vehicle USD declares, and the example controllers self-assemble beside their flight
scripts in ``nexus_sim/examples/controllers/*/assembly.py``, reusing the helpers here,
``build_scenario`` / ``resolve_device``.

Rendering enters through one seam, injected per build: ``renderer_factory``, from
:func:`~nexus_sim._src.rendering.rtx_renderer`, or ``None``, which renders nothing. After the physics
build, since the render poses the stage prims from the model's bodies, its ``link`` gives the render
link, the sensors build in one pass, each RTX sensor over that link, and its ``finish`` hands the link
those sensors.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import warp as wp

from nexus_sim._src.core import Clock, Orchestrator, SeedTree, logger
from nexus_sim._src.physics import NewtonPhysics
from nexus_sim._src.scene import Site


@dataclass(slots=True)
class Assembly:
    """The shared component bundle: everything but the run loop."""

    clock: Any
    physics: Any
    commands: list  # the command stages: the rotors'
    forces: list  # the force elements: the propellers'
    sensors: list
    controller: Any
    logger: Any  # the Rerun sink, or None for no recording


def rotor_chain(physics, vehicle_usd) -> tuple[list, list]:
    """The shipped rotor chain from the rotors the vehicle USD declares, the single, hash-pinned source:
    the rotors' command stage and the propellers' force element, as the command stages and force elements
    a loop takes. The motors between them are Newton's, ``NewtonActuator`` prims the USD authors on the
    rotor joints, a ControllerPID velocity servo + the ClampingDCMotor envelope each, which physics steps
    before its solver, so the rotor speed is a physical joint state: real motor lag + saturation, and
    spinning props at no extra cost.
    """
    from nexus_sim._src.vehicle.commands import RotorCommand
    from nexus_sim._src.vehicle.forces import Propellers

    values = vehicle_usd.actuator_params()
    joints = vehicle_usd.rotor_joints()
    command = RotorCommand(
        model=physics.model, control=physics.control, joints=joints, ct=values["ct"], cd=values["cd"],
        rpm_max=values["rpm_max"],
    )  # fmt: skip
    propellers = Propellers(
        model=physics.model, joints=joints, ct=values["ct"], cd=values["cd"],
        aero_h=values.get("aero_h", 0.0),  # forward-flight thrust loss; 0 = quasi-static kf·Ω²
        aero_hforce=values.get("aero_hforce", 0.0),  # in-plane H-force, drag
    )  # fmt: skip
    return [command], [propellers]


def assemble(
    physics,
    vehicle_usd,
    cfg: dict,
    *,
    controller,
    rerun: bool = False,
    viewer: bool = True,
    debug: bool = False,
    link=None,
    components=None,
    settings: dict | None = None,
) -> Assembly:
    """Assemble the shared core components around the constructed ``physics`` and the
    caller-supplied ``controller``; which controller flies is the launch layer's decision.
    ``link`` is the render link a sensor whose class requires the Kit peer takes, and ``components``
    the registry that resolves each sensor's schema, ``None`` for the default one.
    ``settings`` is the run's effective configuration for the viewer's Settings tab; the launch
    glue passes the tested-config receipt.
    """
    from nexus_sim._src.vehicle.sensors.declared import build_sensors, sensor_specs

    dt = cfg["physics"]["dt"]
    rtf = cfg["physics"].get("rtf", 0)
    # The site: the scene's geodetic origin, which the launch glue threaded into the cfg, and the
    # ambient values resolved from it once, here, for the sensors that read them.
    gps = cfg["sensors"]["gps"]["init"]
    site = Site.at(gps["lat"], gps["lon"], gps["alt"])
    commands, forces = rotor_chain(physics, vehicle_usd)  # from the vehicle USD's rotors, not cfg

    seedtree = SeedTree(cfg.get("seed", 42))  # launch glue threads runtime.seed; string-name path keeps 42
    # The sensors come from the schemas the vehicle USD applies, the single, hash-pinned source, the
    # same as the preceding rotor chain; only the site stays config, since it's a world property,
    # not a vehicle one. A vehicle that declares no sensor builds: which sensors a flight needs is its
    # controller's matter.
    usd_path = vehicle_usd.cfg["usd_path"]
    sensors = build_sensors(
        sensor_specs(usd_path, components),
        usd_path=usd_path,
        model=physics.model,
        seedtree=seedtree,
        dt=dt,
        site=site,
        link=link,
    )
    # The central Rerun recording, §10: built here because it needs the physics Model; the import is
    # lazy so rerun is only pulled in when logging is on. Resilient: a logging stack that fails to
    # build must never block the flight; warn and fly without the sink.
    sink = None
    if rerun:
        try:
            from nexus_sim._src.logging import build_logger

            sink = build_logger(physics.model, viewer=viewer, debug=debug, settings=settings)
        except Exception as exc:
            from nexus_sim._src.core import logger

            logger.warning(f"Rerun recording disabled (logging stack unavailable): {exc}")

    return Assembly(
        clock=Clock(dt, rtf=rtf),
        physics=physics,
        commands=commands,
        forces=forces,
        sensors=sensors,
        controller=controller,
        logger=sink,
    )


# Default scenario; its Global Positioning System (GPS) origin is Seattle.
DEFAULT_SCENARIO = {
    "physics": {"enabled": True, "dt": 0.004, "force_cpu": False, "rtf": 0, "solver": "mujoco"},
    # sensors.gps.init is the scene's geodetic origin, a world property, so it stays config. The
    # sensor suite itself, which sensors exist + their noise/mount params, lives in the vehicle
    # USD as applied sensor schemas, the same as the actuator entry below.
    "sensors": {
        "gps": {"init": {"lat": 47.747944, "lon": -122.163917, "alt": 5.02}},
    },
    # No rotor entry, by design. The aero/thrust map (rpm_max, ct, cd, tau, aero_h, aero_hforce)
    # lives in the vehicle USD, declared per rotor on its body, its motor and its joint, and
    # VehicleUsd.actuator_params() reads it; since the USD is content-hashed, the vehicle hash pins the
    # rotor chain, not any config. Every build reads those params directly from the builder.
}


def build_scenario() -> dict:
    return copy.deepcopy(DEFAULT_SCENARIO)


def resolve_device(cfg: dict) -> str:
    """Select + activate the Warp device for this build, returning its name. ``physics.force_cpu``
    forces CPU, the bit-exact determinism authority; otherwise prefer CUDA when available so
    the live sim runs on the GPU and the captured strategy can engage, falling
    back to CPU when there is no CUDA device. Must run *before* ``NewtonPhysics`` builds the model, as
    the model + state live on the active device. Newton's own default device is CPU, so a non-CPU
    run must set CUDA explicitly here, not rely on the Warp default.
    """
    if cfg["physics"].get("force_cpu") or not wp.is_cuda_available():
        wp.set_device("cpu")
        return "cpu"
    wp.set_device("cuda:0")
    return "cuda:0"


def build_orchestrator(
    vehicle: str,
    cfg: dict,
    vehicle_usd=None,
    *,
    controller,
    rerun: bool = False,
    viewer: bool = True,
    debug: bool = False,
    renderer_factory=None,
    peers=(),
    ports=None,
    preroll_timeout: float = 30.0,
    max_steps: int | None = None,
    settings: dict | None = None,
    components=None,
) -> Orchestrator:
    """The core orchestrator: the one ``NewtonPhysics`` + the one shared assembly around the
    caller-supplied ``controller``; a run differs only in its renderer.

    ``renderer_factory`` is the one rendering seam, from :func:`~nexus_sim._src.rendering.rtx_renderer`:
    used after the physics build, since the render poses stage prims from the model's bodies;
    ``None`` renders nothing. ``components`` resolves each sensor's schema to its class.
    ``peers`` are the processes the build started for this run, which the loop stops when the run ends.
    ``ports`` is the run's port map, the links that leave the run, which ``Sim.ports`` exposes.
    ``settings`` is the run's effective configuration, which the loop keeps for a caller to read back
    and the logger shows in the viewer's Settings tab; the launch glue passes the tested-config receipt.
    ``preroll_timeout`` covers a host-boundary controller's boot, since an autopilot in a container
    needs a generous window.
    """
    logger.info(f"device: {resolve_device(cfg)}")
    if vehicle_usd is None:
        raise ValueError(f"vehicle_usd is required for {vehicle!r} (resolve it via the launch glue)")
    physics = NewtonPhysics(vehicle_usd=vehicle_usd, cfg=cfg)
    renderer = renderer_factory.link(physics, vehicle_usd) if renderer_factory else None
    a = assemble(
        physics, vehicle_usd, cfg,
        controller=controller, rerun=rerun, viewer=viewer, debug=debug, link=renderer, components=components,
        settings=settings,
    )  # fmt: skip
    if renderer_factory:
        rtx = [s for s in a.sensors if getattr(s, "requires", ())]
        renderer_factory.finish(renderer, rtx, vehicle_usd, cfg)
    return Orchestrator(
        clock=a.clock,
        physics=a.physics,
        commands=a.commands,
        forces=a.forces,
        sensors=a.sensors,
        controller=a.controller,
        logger=a.logger,
        renderer=renderer,
        peers=peers,
        ports=ports,
        preroll_timeout=preroll_timeout,
        max_steps=max_steps,
        settings=settings,
    )
