"""The core orchestrator assembly, shared by every runtime, per §12.

Controller-agnostic by construction: the assembly wires the plant, ``NewtonPhysics`` from nexus's
own ModelBuilder + solvers, the shipped ``ArticulatedRotors`` actuator, the sensor suite authored in
Universal Scene Description (USD), the environment, and the Rerun sink around a **caller-supplied
controller**. Which controller flies is a decision one layer up: the launch glue maps the config's
``control.kind``, and the example controllers self-assemble beside their flight scripts in
``nexus/examples/controllers/*/assembly.py``, reusing the helpers here,
``build_scenario`` / ``resolve_device``.

No component binds to a runtime either: the same assembly runs identically in either host, the
headless process or the Isaac Sim Kit app. A runtime differs in exactly one seam, injected per build::

    renderer_factory(physics, vehicle_builder, cfg) -> (renderer, extra_sensors)

``None``, the headless default, renders nothing; the Isaac runtime passes a factory that stands up
the RtxFrame + the vehicle USD's authored RTX camera/lidar sensors. Called after the physics build,
since a renderer maps its stage prims onto the model's bodies.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import warp as wp

from nexus._src.core import Clock, ConstantEnvironment, Orchestrator, SeedTree, logger
from nexus._src.physics import NewtonPhysics
from nexus._src.vehicle.actuators import check_actuator_model_pairing


@dataclass(slots=True)
class Assembly:
    """The shared component bundle: everything but the run loop."""

    clock: Any
    environment: Any
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
    """Assemble the shared core components around the runtime-constructed ``physics`` and the
    caller-supplied ``controller``; which controller flies is the launch layer's decision.
    ``settings`` is the run's effective configuration for the viewer's Settings tab; the launch
    glue passes the tested-config receipt.
    """
    from nexus._src.vehicle.actuators import ArticulatedRotors
    from nexus._src.vehicle.sensors.usd import build_sensors

    dt = cfg["physics"]["dt"]
    rtf = cfg["physics"].get("rtf", 0)
    gps = cfg["sensors"]["gps"]["init"]
    # Actuator aero/thrust map: read straight from the motor:*/propeller:* attrs of the vehicle USD, the
    # single, hash-pinned source. Not in cfg.
    act = vehicle_builder.actuator_params()
    # A scene's geodetic_origin.alt, if any, already landed in the Global Positioning System (GPS) init.
    ref_alt = gps["alt"]

    # The shipped actuator: motors as USD-authored ``newton.actuators``, NewtonActuator prims,
    # a ControllerPID velocity servo + the ClampingDCMotor envelope on each real actuator joint, and aero,
    # thrust/H-force from the solver-integrated Ω via the propeller:* attrs, as nexus's body_f kernel.
    # Ω is a physical joint state: real motor lag + saturation, and spinning props at no extra cost.
    actuator = ArticulatedRotors(
        model=physics.model,
        control=physics.control,
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
    # source, the same as the preceding actuator params; only the geodetic origin stays config, since it
    # is a world property, not a vehicle one. A controller flying Hardware In The Loop (HIL) is dead
    # without sensors, so an unauthored USD fails loudly here.
    specs = vehicle_builder.sensor_specs()
    if not specs:
        raise ValueError(
            "no sensor:* prims authored in the vehicle USD: this assembly builds the USD-authored "
            "analytic sensor suite, so author it as sensor:* prims under the base body"
        )
    sensors = [
        *build_sensors(specs, seedtree=seedtree, dt=dt, gps_init=gps, ref_alt=ref_alt),
        *(extra_sensors or []),  # runtime extras, for example USD-discovered RTX camera sensors, host-rate
    ]
    # World Magnetic Model (WMM) field at the GPS origin, the autopilot's own coarse table, so strict mag
    # arming checks pass.
    environment = ConstantEnvironment.from_gps(gps["lat"], gps["lon"])

    # The central Rerun recording, §10: built here because it needs the physics Model; the import is
    # lazy so rerun is only pulled in when logging is on. Resilient on the Isaac path: a missing rerun
    # stack in the Kit env must never block the flight; warn and fly without the sink.
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
        environment=environment,
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
    # lives in the vehicle USD as motor:* / propeller:* custom attributes on the actuator joint prims, and
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
    preroll_timeout: float = 30.0,
    max_steps: int | None = None,
    settings: dict | None = None,
) -> Orchestrator:
    """The core orchestrator: the one ``NewtonPhysics`` + the one shared assembly around the
    caller-supplied ``controller``; a runtime differs only in its renderer, architecture.md §12.

    ``renderer_factory(physics, vehicle_builder, cfg) -> (renderer, extra_sensors)`` is the one
    runtime seam: called after the physics build, since the renderer maps stage prims onto model
    bodies; ``None``, the headless default, renders nothing.
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
        environment=a.environment,
        physics=a.physics,
        actuator=a.actuator,
        sensors=a.sensors,
        controller=a.controller,
        logger=a.logger,
        renderer=renderer,
        preroll_timeout=preroll_timeout,
        max_steps=max_steps,
    )


def run_captured(orch: Orchestrator, *, steps: int) -> Orchestrator:
    """Thin wrapper that runs ``orch`` under the in-process captured strategy for a fixed ``steps``.
    The strategy itself now lives in :meth:`Orchestrator._loop_captured_inprocess`; ``run()`` selects
    it automatically. Kept for the explicit bounded-replay use, tests and benchmarks. Requires CUDA;
    falls back to eager ``run()`` on CPU.
    """
    import warp as wp

    if not wp.get_device().is_cuda:
        orch.run()  # capture needs CUDA; eager is the CPU path
        return orch
    state = orch.physics.reset()
    orch.controller.connect()
    try:
        for _ in orch._loop_captured_inprocess(state, steps=steps):  # generator; exhaust it, bounded by steps
            pass
    finally:
        orch.controller.close()
        orch._close_logs()
    return orch


def run_captured_host_exchange(orch: Orchestrator, *, steps: int | None = None, rtf: float = 0.0) -> Orchestrator:
    """Thin wrapper that runs ``orch`` under the host-exchange captured strategy. The strategy itself
    now lives in :meth:`Orchestrator._loop_captured_host_exchange`; ``run()`` selects it automatically
    for a non-capturable controller on CUDA. Kept for explicit/throttled invocation: ``rtf=0``, the
    default, runs as fast as the controller keeps up, ``rtf=1.0`` throttles to real-time for interactive
    flying; ``steps=None`` runs until the controller disconnects. Requires CUDA; falls back to eager
    ``run()`` on CPU.

    **Validated end-to-end against a real host-boundary autopilot**, PX4 Software In The Loop (SITL),
    on an RTX 5080 with ``none_astro_max``: arms + climbs via the ``px4_sitl`` example's flight script;
    capture keeps the lockstep loop fed far faster than the eager per-tick Python path, and lockstep
    doesn't cap it near real-time. The sensors' in-graph step counter covers the load-bearing condition:
    sensor noise must dither per replay, or the autopilot's Extended Kalman Filter (EKF) detects a stuck
    sensor and won't arm; see :meth:`Orchestrator._loop_captured_host_exchange`.
    """
    import warp as wp

    if not wp.get_device().is_cuda:
        orch.run()  # capture needs CUDA
        return orch
    orch.clock.rtf = rtf  # rtf=0, the default: run as fast as the controller keeps up; rtf>0 throttles
    state = orch.physics.reset()
    try:
        # The loop connects the controller itself, after its capture: a host-boundary controller
        # dials in there, for example PX4 on tcpin:4560.
        for _ in orch._loop_captured_host_exchange(state, steps=steps):  # generator; exhaust it
            pass
    except ConnectionError as e:
        logger.info(str(e))
    finally:
        orch.controller.close()
        orch._close_logs()
    return orch
