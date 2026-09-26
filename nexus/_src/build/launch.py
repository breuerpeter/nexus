"""Launch glue: a ``LaunchConfig`` → a running ``Orchestrator``, the one path every run builds through.

Resolves the launch against the registry, sha-verifying every asset, wraps the vehicle
Universal Scene Description (USD) file in a :class:`USDBuilder`, threads the resolved scene, its
USD plus start plus geodetic origin, into the scenario cfg, and assembles the core orchestrator,
the controller-agnostic ``build.assembly``, around the controller the config declares. PX4 is
the one first-class control kind, and *this* layer turns ``control.kind`` into the
``Px4MavlinkController`` instance; every other controller is an example that self-assembles its
orchestrator via :func:`resolve_scenario` plus ``Sim.from_orchestrator``, and renders through
:func:`~nexus._src.rendering.rtx_renderer` the same way.

A vehicle whose USD authors RTX sensor prims renders them in the Kit peer, a container this build
starts right after the fetch, so Kit boots while PX4 builds and the physics compiles.
"""

from __future__ import annotations

import pathlib

from nexus._src.config import LaunchConfig, Registry, ResolvedLaunch, resolve
from nexus._src.core import Orchestrator
from nexus._src.physics import USDBuilder
from nexus._src.rendering import rtx_renderer

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
    """Thread the resolved scene into the cfg, uniformly, for every run.

    The physics model build loads the scene USD plus its start point exactly the same way as the
    vehicle USD: physics takes the UsdPhysics-authored prims, see ``scene.add_scene``. When the
    vehicle renders, the Kit peer opens the same file as its render stage and finds by its
    content whether it needs live machinery, for example the cesium globe. A geolocated scene anchors
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


def build_from_launch(
    launch: LaunchConfig,
    *,
    registry: Registry | None = None,
    cache_dir: str | pathlib.Path | None = None,
    cfg: dict | None = None,
    stream: bool = False,
    preroll_timeout: float = 30.0,
) -> Orchestrator:
    """Resolve *launch* and assemble the core PX4 Orchestrator.

    PX4 is the one first-class control kind; every other controller is an example that
    self-assembles from ``resolve_scenario`` plus its own components plus ``Sim.from_orchestrator``.
    ``stream`` publishes each RTX camera's feed over Real Time Streaming Protocol (RTSP).

    Raises:
        KitPeerError: The vehicle authors RTX sensors and the Kit peer couldn't start.
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
    # By now the run has fetched every asset it renders, so the Kit peer starts first and boots while
    # PX4 builds and the physics compiles; None for a vehicle with no RTX sensor prims.
    renderer_factory = rtx_renderer(builder, cfg, cache_dir=cache_dir, stream=stream)
    try:
        # *This* is where the declared control kind becomes a controller instance; the assembly that
        # follows is controller-agnostic. Lazy import: only the PX4 path pulls in pymavlink.
        from nexus._src.vehicle.controllers.px4 import Px4MavlinkController

        spec = resolved.tested_config.px4
        controller = Px4MavlinkController(**({"airframe": spec.airframe} if spec is not None else {}))
        # Blocking: the PX4 prerequisites plus its incremental build, before any preroll clock
        # starts, so `Sim`, `nexus run` and the benchmark cell all get the PX4 lifecycle from here.
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
    except BaseException:
        if renderer_factory is not None:
            renderer_factory.close()  # the loop never took the peer over, so its container stops here
        raise
