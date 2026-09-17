"""Launch glue: a ``newton-config`` ``LaunchConfig`` → a running ``Orchestrator``, one glue for
every runtime.

Resolves the launch against the registry, sha-verifying every asset, wraps the vehicle
Universal Scene Description (USD) file in a :class:`USDBuilder`, threads the resolved scene, its
USD plus start plus geodetic origin, into the scenario cfg, and assembles the core orchestrator,
the controller-agnostic ``runtimes.assembly``, around the controller the config declares. PX4 is
the one first-class control kind, and *this* layer turns ``control.kind`` into the
``Px4MavlinkController`` instance; every other controller is an example that self-assembles its
orchestrator via :func:`resolve_scenario` plus ``Sim.from_orchestrator``. The launch surface is
runtime-agnostic by construction: a runtime enters *only* through the two injection points,
``renderer_factory`` and ``preroll_timeout``, so any orchestrator setup runs identically wherever
it launches. ``renderer_factory`` supplies the Isaac RtxFrame plus RTX sensors; ``None`` renders
nothing, and :func:`default_renderer_factory` resolves it from the booted app.
"""

from __future__ import annotations

import pathlib

from nexus._src.config import LaunchConfig, Registry, ResolvedLaunch, resolve
from nexus._src.core import Orchestrator
from nexus._src.physics import USDBuilder

from .assembly import build_orchestrator, build_scenario


def resolve_to_vehicle_builder(
    launch: LaunchConfig,
    registry: Registry | None = None,
    *,
    cache_dir: str | pathlib.Path | None = None,
) -> tuple[USDBuilder, ResolvedLaunch]:
    """Resolve *launch*, fetching and sha-verifying assets, and wrap the vehicle USD in a USDBuilder."""
    resolved = resolve(launch, registry, cache_dir=cache_dir)
    builder = USDBuilder({"usd_path": resolved.vehicle_usd_path}, None)
    return builder, resolved


def _scenario_from_launch(launch: LaunchConfig) -> dict:
    """Map the LaunchConfig runtime fields onto the scenario cfg dict.

    Honors dt, device, cpu or cuda, and seed. Still TODO, as a follow-on: the GPU *ordinal*, since
    only cpu-or-cuda threads through, not cuda:1; ``substeps``, since the policy path intentionally
    pins physics_substeps=1 for thrust calibration; and sensor overrides. So the tested-config receipt
    is faithful for what's mapped here; nothing consumes the unmapped runtime fields yet.
    """
    cfg = build_scenario()
    rt = launch.runtime
    cfg["physics"]["dt"] = rt.dt
    # Only an *explicit* "cpu" forces CPU; "auto" and "cuda:*" let resolve_device pick CUDA when available.
    # The GPU ordinal is still TODO: only cpu-or-cuda threads through.
    cfg["physics"]["force_cpu"] = rt.device == "cpu"
    cfg["physics"]["solver"] = rt.solver  # mujoco, the default | semi_implicit | featherstone
    cfg["physics"]["rtf"] = rt.rtf  # 0 = unthrottled; 1.0 = pace to wall-clock, for interactive flying
    cfg["seed"] = rt.seed  # honored by build_orchestrator's SeedTree
    return cfg


def _thread_scene(cfg: dict, resolved: ResolvedLaunch) -> None:
    """Thread the resolved scene into the cfg, uniformly, for every runtime.

    The physics model build loads the scene USD plus its start point exactly the same way as the
    vehicle USD: physics takes the UsdPhysics-authored prims, see ``scene.add_scene``. Where
    a renderer exists, the build also references the scene onto the render stage, and the RtxFrame
    finds a scene *handler* by the path, for example the cesium globe. A geolocated scene anchors
    *both* the render world, ``rtx.georef``, and the Hardware In The Loop (HIL) Global Positioning
    System (GPS), ``sensors.gps.init``, at its geodetic origin: the scene is the single source of
    the "where on Earth" answer, so the Ground Control Station (GCS) minimap and the camera feed agree.
    """
    cfg["scene_usd_path"] = resolved.scene_usd_path
    cfg["scene_start"] = resolved.tested_config.scene_start
    go = resolved.tested_config.geodetic_origin
    if go is not None:
        cfg["rtx"] = {**cfg.get("rtx", {}), "georef": {"lat": go.lat, "lon": go.lon, "alt": go.alt}}
        sensors = cfg.get("sensors", {})
        gps = sensors.get("gps", {})
        init = {**gps.get("init", {}), "lat": go.lat, "lon": go.lon}
        if go.alt is not None:
            init["alt"] = go.alt
        cfg["sensors"] = {**sensors, "gps": {**gps, "init": init}}


def resolve_scenario(
    launch: LaunchConfig,
    *,
    registry: Registry | None = None,
    cache_dir: str | pathlib.Path | None = None,
) -> tuple[USDBuilder, ResolvedLaunch, dict]:
    """Resolve *launch* to ``(vehicle_builder, resolved, scenario cfg)``: the shared front half of
    any orchestrator assembly. ``build_from_launch`` composes it with the PX4 assembly; a
    self-assembled example, with its own controller plus actuator plus physics, starts from this and
    hands the built orchestrator to ``Sim.from_orchestrator``.
    """
    builder, resolved = resolve_to_vehicle_builder(launch, registry, cache_dir=cache_dir)
    cfg = _scenario_from_launch(launch)
    _thread_scene(cfg, resolved)
    return builder, resolved, cfg


def default_renderer_factory(stream: bool = False):
    """The runtime's renderer seam for self-assembled orchestrators: the Isaac RTX factory when a
    Kit app has booted in *this* process, the ``nexus script`` path, else ``None`` for headless.
    Detection is by the booted app, not importability: ``import isaacsim`` exists but is unusable
    before ``SimulationApp`` starts.
    """
    import sys

    if "isaacsim" in sys.modules and "omni.kit.app" in sys.modules:
        from nexus._src.runtimes.isaacsim.launch import _renderer_factory

        return _renderer_factory(stream=stream)
    return None


def build_from_launch(
    launch: LaunchConfig,
    *,
    registry: Registry | None = None,
    cache_dir: str | pathlib.Path | None = None,
    cfg: dict | None = None,
    renderer_factory=None,
    preroll_timeout: float = 30.0,
) -> Orchestrator:
    """Resolve *launch* and assemble the core PX4 Orchestrator.

    ``renderer_factory`` and ``preroll_timeout`` are the *only* runtime injections; see the module
    docstring. PX4 is the one first-class control kind; every other controller is an example that
    self-assembles from ``resolve_scenario`` plus its own components plus ``Sim.from_orchestrator``.
    """
    builder, resolved = resolve_to_vehicle_builder(launch, registry, cache_dir=cache_dir)
    if cfg is None:
        cfg = _scenario_from_launch(launch)
    _thread_scene(cfg, resolved)
    label = resolved.tested_config.vehicle or "vehicle"
    kind = launch.control.kind
    if kind != "px4-sitl":
        raise ValueError(
            f"control.kind {kind!r} is not a core control kind (PX4 is the one first-class controller); "
            "other controllers live in nexus/examples/ and self-assemble via Sim.from_orchestrator"
        )
    # *This* is where the declared control kind becomes a controller instance; the assembly that
    # follows is controller-agnostic. Lazy import: only the PX4 path pulls in pymavlink.
    from nexus._src.vehicle.controllers.px4 import Px4MavlinkController

    spec = resolved.tested_config.px4
    controller = Px4MavlinkController(**({"airframe": spec.airframe} if spec is not None else {}))
    # Blocking: the PX4 prerequisites plus its incremental build, before any preroll clock starts. This
    # is the seam *both* runtimes share, so the standalone front door, Sim, `nexus run` on either
    # runtime, `nexus script` and the benchmark cell all get the PX4 lifecycle from here.
    controller.prepare()

    # output.log/view → the Logger, which writes the .rrd or serves :9876; neither → no recording.
    return build_orchestrator(
        label,
        cfg,
        vehicle_builder=builder,
        controller=controller,
        rerun=launch.output.log or launch.output.view,
        viewer=launch.output.view,
        debug=launch.output.debug,
        renderer_factory=renderer_factory,
        preroll_timeout=preroll_timeout,
        max_steps=launch.runtime.max_steps,
        # The viewer's Settings tab shows the tested-config receipt, "every input that affects the
        # simulation" per config.receipt, so a recording says what produced it.
        settings=resolved.tested_config.model_dump(mode="json"),
    )


def attach_default_recorder(orch: Orchestrator, launch: LaunchConfig) -> None:
    """Attach the observation ``Recorder`` a launch-driven run gets by default: the same default the
    ``Sim`` front door applies with ``observe=True``, so a command-line flight records every component
    channel and the Logger's end-of-run debug dump has rings to read. Sized to cover the whole run,
    mirroring ``Sim.__enter__``; must run *before* ``run()``, since a captured graph records the taps
    at capture time.
    """
    from nexus._src.recording import Recorder

    max_steps = launch.runtime.max_steps
    maxlen = max(4096, int(max_steps) + 64) if max_steps else 30_000  # ~2 min @ 250 Hz when unbounded
    orch.attach_recorder(Recorder(dt=launch.runtime.dt, maxlen=maxlen))


def run_from_launch(launch: LaunchConfig, **kwargs) -> Orchestrator:
    """Resolve, assemble, and run a launch through ``Orchestrator.run()``. Returns the orchestrator."""
    orch = build_from_launch(launch, **kwargs)
    attach_default_recorder(orch, launch)
    orch.run()
    return orch
