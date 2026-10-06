"""The core orchestrator assembly, shared by every run, per §12.

Controller-agnostic by construction: the assembly wires the plant, ``NewtonPhysics`` from nexus's
own ModelBuilder + solvers, the shipped ``ArticulatedRotors`` actuator, the sensor suite authored in
Universal Scene Description (USD) with the site's ambient values, and the Rerun sink around a
**caller-supplied controller**. Which controller flies is a decision one layer up: the launch glue
builds the controller the vehicle USD declares, and the example controllers self-assemble beside their flight
scripts in ``nexus/examples/controllers/*/assembly.py``, reusing the helpers here,
``build_scenario`` / ``resolve_device``.

Rendering enters through one seam, injected per build::

    renderer_factory(physics, vehicle_builder, cfg) -> (renderer, extra_sensors)

``None`` renders nothing; :func:`~nexus._src.rendering.rtx_renderer` returns a factory that maps the
model's bodies onto the Kit peer's render stage and builds the vehicle USD's authored RTX
camera/lidar sensors over the render link. Called after the physics build, since the render poses
the stage prims from the model's bodies.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import warp as wp

from nexus._src.core import Clock, Orchestrator, SeedTree, logger
from nexus._src.physics import NewtonPhysics
from nexus._src.scene import Site
from nexus._src.vehicle.actuators import check_actuator_model_pairing


@dataclass(slots=True)
class Assembly:
    """The shared component bundle: everything but the run loop."""

    clock: Any
    physics: Any
    actuator: Any
    sensors: list
    controller: Any
    logger: Any  # the Rerun sink, or None for no recording


def assemble(
    physics,
    vehicle_builder,
    cfg: dict,
    *,
    controller,
    rerun: bool = False,
    viewer: bool = True,
    debug: bool = False,
    extra_sensors: list | None = None,
    settings: dict | None = None,
) -> Assembly:
    """Assemble the shared core components around the constructed ``physics`` and the
    caller-supplied ``controller``; which controller flies is the launch layer's decision.
    ``settings`` is the run's effective configuration for the viewer's Settings tab; the launch
    glue passes the tested-config receipt.
    """
    from nexus._src.vehicle.actuators import ArticulatedRotors
    from nexus._src.vehicle.sensors.usd import build_sensors

    dt = cfg["physics"]["dt"]
    rtf = cfg["physics"].get("rtf", 0)
    # The site: the scene's geodetic origin, which the launch glue threaded into the cfg, and the
    # ambient values resolved from it once, here, for the sensors that read them.
    gps = cfg["sensors"]["gps"]["init"]
    site = Site.at(gps["lat"], gps["lon"], gps["alt"])
    # Actuator aero/thrust map: read straight from the rotors the vehicle USD declares, the single,
    # hash-pinned source. Not in cfg.
    act = vehicle_builder.actuator_params()
    # The shipped actuator: motors as USD-authored ``newton.actuators``, NewtonActuator prims,
    # a ControllerPID velocity servo + the ClampingDCMotor envelope on each real actuator joint, and aero,
    # thrust/H-force from the solver-integrated Ω via each rotor body's propeller schema, as nexus's
    # body_f kernel. Ω is a physical joint state: real motor lag + saturation, and spinning props at no
    # extra cost.
    actuator = ArticulatedRotors(
        model=physics.model,
        control=physics.control,
        joints=vehicle_builder.rotor_joints(),
        ct=act["ct"],
        cd=act["cd"],
        rpm_max=act["rpm_max"],
        dt=dt,
        aero_h=act.get("aero_h", 0.0),  # forward-flight thrust loss; 0 = quasi-static kf·Ω²
        aero_hforce=act.get("aero_hforce", 0.0),  # in-plane H-force, drag
    )
    check_actuator_model_pairing(actuator, physics.model)  # requires the USD-authored motors

    seedtree = SeedTree(cfg.get("seed", 42))  # launch glue threads runtime.seed; string-name path keeps 42
    # The analytic sensor suite comes from the sensor:* prims of the vehicle USD, the single, hash-pinned
    # source, the same as the preceding actuator params; only the site stays config, since it is a
    # world property, not a vehicle one. A controller flying Hardware In The Loop (HIL) is dead
    # without sensors, so an unauthored USD fails loudly here.
    specs = vehicle_builder.sensor_specs()
    if not specs:
        raise ValueError(
            "no sensor:* prims authored in the vehicle USD: this assembly builds the USD-authored "
            "analytic sensor suite, so author it as sensor:* prims under the base body"
        )
    sensors = [
        *build_sensors(specs, seedtree=seedtree, dt=dt, site=site),
        *(extra_sensors or []),  # the renderer's sensors, for example USD-discovered RTX cameras, host-rate
    ]
    # The central Rerun recording, §10: built here because it needs the physics Model; the import is
    # lazy so rerun is only pulled in when logging is on. Resilient: a logging stack that fails to
    # build must never block the flight; warn and fly without the sink.
    sink = None
    if rerun:
        try:
            from nexus._src.logging import build_logger

            sink = build_logger(physics.model, viewer=viewer, debug=debug, settings=settings)
        except Exception as exc:
            from nexus._src.core import logger

            logger.warning(f"Rerun recording disabled (logging stack unavailable): {exc}")

    return Assembly(
        clock=Clock(dt, rtf=rtf),
        physics=physics,
        actuator=actuator,
        sensors=sensors,
        controller=controller,
        logger=sink,
    )


# Default scenario: mirrors the bridge config.yaml; GPS origin = Seattle.
DEFAULT_SCENARIO = {
    "physics": {"enabled": True, "dt": 0.004, "force_cpu": False, "rtf": 0, "solver": "mujoco"},
    # sensors.gps.init is the scene's geodetic origin, a world property, so it stays config. The
    # sensor suite itself, which sensors exist + their noise/mount params, lives in the vehicle
    # USD as sensor:* prims, and USDBuilder.sensor_specs() reads it, the same as the actuator entry below.
    "sensors": {
        "gps": {"init": {"lat": 47.747944, "lon": -122.163917, "alt": 5.02}},
    },
    # No actuator entry, by design. The aero/thrust map (rpm_max, ct, cd, tau, aero_h, aero_hforce)
    # lives in the vehicle USD, declared per rotor on its body, its motor and its joint, and
    # USDBuilder.actuator_params() reads it; since the USD is content-hashed, the vehicle hash pins the
    # actuator model, not any config. Every build reads those params directly from the builder.
}


def build_scenario() -> dict:
    return copy.deepcopy(DEFAULT_SCENARIO)


def resolve_device(cfg: dict) -> str:
    """Select + activate the Warp device for this build, returning its name. ``physics.force_cpu``
    forces CPU, the bit-exact determinism authority; otherwise prefer CUDA when available so
    the live sim runs on the GPU and the captured strategy, architecture.md §5, can engage, falling
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
    vehicle_builder=None,
    *,
    controller,
    rerun: bool = False,
    viewer: bool = True,
    debug: bool = False,
    renderer_factory=None,
    peers=(),
    preroll_timeout: float = 30.0,
    max_steps: int | None = None,
    settings: dict | None = None,
) -> Orchestrator:
    """The core orchestrator: the one ``NewtonPhysics`` + the one shared assembly around the
    caller-supplied ``controller``; a run differs only in its renderer, architecture.md §12.

    ``renderer_factory(physics, vehicle_builder, cfg) -> (renderer, extra_sensors)`` is the one
    rendering seam: called after the physics build, since the render poses stage prims from the
    model's bodies; ``None`` renders nothing.
    ``peers`` are the processes the build started for this run, which the loop stops when the run ends.
    ``settings`` is the run's effective configuration, which the loop keeps for a caller to read back
    and the logger shows in the viewer's Settings tab; the launch glue passes the tested-config receipt.
    ``preroll_timeout`` covers a host-boundary controller's boot, since an autopilot in a container
    needs a generous window.
    """
    logger.info(f"device: {resolve_device(cfg)}")
    if vehicle_builder is None:
        raise ValueError(f"vehicle_builder is required for {vehicle!r} (resolve it via the launch glue)")
    physics = NewtonPhysics(vehicle_builder=vehicle_builder, cfg=cfg)
    renderer, extra_sensors = renderer_factory(physics, vehicle_builder, cfg) if renderer_factory else (None, [])
    a = assemble(
        physics, vehicle_builder, cfg,
        controller=controller, rerun=rerun, viewer=viewer, debug=debug, extra_sensors=extra_sensors,
        settings=settings,
    )  # fmt: skip
    return Orchestrator(
        clock=a.clock,
        physics=a.physics,
        actuator=a.actuator,
        sensors=a.sensors,
        controller=a.controller,
        logger=a.logger,
        renderer=renderer,
        peers=peers,
        preroll_timeout=preroll_timeout,
        max_steps=max_steps,
        settings=settings,
    )
