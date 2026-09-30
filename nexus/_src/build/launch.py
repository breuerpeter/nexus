"""Launch glue: a ``LaunchConfig`` → a running ``Orchestrator``, the one path every run builds through.

Resolves the launch against the registry, sha-verifying every asset, wraps the vehicle
Universal Scene Description (USD) file in a :class:`USDBuilder`, threads the resolved scene, its
USD plus start plus geodetic origin, into the scenario cfg, and assembles the core orchestrator,
the controller-agnostic ``build.assembly``, around the one controller the vehicle USD declares on
its root prim. PX4 is the one first-class controller, and *this* layer builds its
``Px4MavlinkController`` from the ``NexusPx4API`` schema; every other controller is an example that
self-assembles its orchestrator via :func:`resolve_scenario` plus ``Sim.from_orchestrator``, and
renders through :func:`~nexus._src.rendering.rtx_renderer` the same way.

A vehicle whose USD authors RTX sensor prims renders them in the Kit peer, a container this build
starts right after the fetch, so Kit boots while PX4 builds and the physics compiles. The PX4
autopilot is a peer too: when the vehicle declares the PX4 Software In The Loop (SITL) peer,
``NexusPx4SitlAPI``, the build starts its container on the peer contract before the assembly, on a
PX4 instance free on this machine, and hands it to the orchestrator, which stops it when the run
ends. A run whose override layer drops that declaration starts nothing, and the controller waits on
instance 0's Hardware In The Loop (HIL) port for an autopilot started elsewhere.

The build starts each peer from the class its peer mapping names: by default the real process, and
in a test the peer's fake, which speaks the same link and starts no process.
"""

from __future__ import annotations

import os
import pathlib
import socket
import time
from collections.abc import Callable, Mapping

from nexus._src.config import LaunchConfig, Px4Spec, Registry, ResolvedLaunch, resolve
from nexus._src.core import Orchestrator
from nexus._src.core.registry import ComponentRegistry
from nexus._src.peers.px4_sitl import HIL_PORT
from nexus._src.physics import USDBuilder
from nexus._src.rendering import rtx_renderer
from nexus._src.usd.reader import read_declarations

from .assembly import build_orchestrator, build_scenario
from .components import declared_controller, root_schemas

# The schema a vehicle declares PX4 with; its `airframe` goes into the receipt.
PX4_SCHEMA = "NexusPx4API"
# The schema a vehicle declares the PX4 SITL peer with, beside PX4_SCHEMA on its root prim.
PX4_SITL_SCHEMA = "NexusPx4SitlAPI"


def shipped_peers() -> dict[str, type]:
    """The peer mapping a run takes by default: each peer's name to the class that starts its real process."""
    from nexus._src.peers.kit.runner import KitPeer
    from nexus._src.peers.px4_sitl.runner import Px4Sitl

    return {"px4_sitl": Px4Sitl, "kit": KitPeer}


def resolve_to_vehicle_builder(
    launch: LaunchConfig,
    registry: Registry | None = None,
    *,
    cache_dir: str | pathlib.Path | None = None,
) -> tuple[USDBuilder, ResolvedLaunch]:
    """Resolve *launch*, fetching and sha-verifying assets, and wrap the vehicle USD in a USDBuilder.

    The receipt records the airframe of the PX4 schema the vehicle declares, if it declares one.
    """
    resolved = resolve(launch, registry, cache_dir=cache_dir)
    if resolved.vehicle_usd_path is not None:
        airframes = [
            kw["airframe"] for _, schema, kw in read_declarations(resolved.vehicle_usd_path) if schema == PX4_SCHEMA
        ]
        if airframes:
            resolved.tested_config.px4 = Px4Spec(airframe=airframes[0])
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
    components: ComponentRegistry | None = None,
    peers: Mapping[str, Callable] | None = None,
) -> Orchestrator:
    """Resolve *launch* and assemble the core Orchestrator around the controller the vehicle declares.

    PX4 is the one first-class controller; every other controller is an example that
    self-assembles from ``resolve_scenario`` plus its own components plus ``Sim.from_orchestrator``.
    ``stream`` publishes each RTX camera's feed over Real Time Streaming Protocol (RTSP).
    ``components`` resolves the vehicle's schemas to classes; ``None`` takes the default registry.
    ``peers`` maps a peer's name, ``px4_sitl`` or ``kit``, to the class the build starts for it, or a
    callable that builds one, over :func:`shipped_peers`: a test sends a peer to its fake here.

    Raises:
        ValueError: The vehicle declares no controller, two, one off its root prim, one other than
            PX4, a PX4 schema with no airframe, or the PX4 SITL peer with no PX4 schema, or ``peers``
            names a peer the build doesn't know; raised before any peer starts.
        FileNotFoundError: The launch names an override layer with no file behind it.
        KitPeerError: The vehicle authors RTX sensors and the Kit peer couldn't start.
    """
    builder, resolved = resolve_to_vehicle_builder(launch, registry, cache_dir=cache_dir)
    if cfg is None:
        cfg = _scenario_from_launch(launch)
    _thread_scene(cfg, resolved)
    label = resolved.tested_config.vehicle or "vehicle"
    shipped = shipped_peers()
    unknown = sorted(set(peers or {}) - set(shipped))
    if unknown:
        raise ValueError(f"the peer mapping names {unknown}, which no peer answers to; the peers are {sorted(shipped)}")
    peer_classes = {**shipped, **(peers or {})}
    root, schemas = root_schemas(resolved.vehicle_usd_path)
    px4_sitl = PX4_SITL_SCHEMA in schemas
    if px4_sitl and PX4_SCHEMA not in schemas:
        raise ValueError(
            f"{root}: {PX4_SITL_SCHEMA} declares the PX4 SITL peer, but the prim declares no {PX4_SCHEMA} to fly it"
        )
    spec = declared_controller(resolved.vehicle_usd_path, components)
    if spec.schema != PX4_SCHEMA:
        raise ValueError(
            f"{spec.prim}: {spec.schema} is not a core controller (PX4 is the one first-class controller); "
            "other controllers live in nexus/examples/ and self-assemble via Sim.from_orchestrator"
        )
    if not spec.kwargs["airframe"]:
        raise ValueError(
            f"{spec.prim}: {PX4_SCHEMA} authors no nexus:airframe; name the PX4 SITL airframe, such as astro_max"
        )
    # By now the run has fetched every asset it renders, so the Kit peer starts first and boots while
    # PX4 builds and the physics compiles; None for a vehicle with no RTX sensor prims.
    renderer_factory = rtx_renderer(builder, cfg, cache_dir=cache_dir, stream=stream, peer=peer_classes["kit"])
    started: list = []
    try:
        # *This* is where the declared controller becomes an instance; the assembly that follows is
        # controller-agnostic. The schema gives its keywords, and the run gives the PX4 peer's addresses.
        instance = _px4_instance(peer_classes["px4_sitl"]) if px4_sitl else 0
        controller = spec.cls(**spec.kwargs, port=HIL_PORT + instance, target_system=instance + 1)
        if px4_sitl:
            # The peer starts here, before the assembly: its start builds PX4 incrementally, which must
            # stay outside the sim's preroll window, GH #39, and PX4 boots while the physics compiles.
            started.append(_start_px4(peer_classes["px4_sitl"], resolved, instance, controller.airframe))

        # output.log/view → the Logger, which writes the .rrd or serves :9876; neither → no recording.
        return build_orchestrator(
            label,
            cfg,
            vehicle_builder=builder,
            controller=controller,
            peers=started,
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
        # The loop never took the peers over, so their containers stop here.
        for peer in started:
            peer.stop()
        if renderer_factory is not None:
            renderer_factory.close()
        raise


def _px4_instance(cls: Callable) -> int:
    """The lowest PX4 instance free on this machine: no live run's container holds it, and nothing
    listens on its HIL port. The run owns every address its peer uses, so it picks the instance itself.
    """
    held = getattr(cls, "held_instances", set)()  # a fake, or a callable that builds one, holds none
    for instance in range(256):
        if instance in held:
            continue
        with socket.socket() as s:
            s.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
            )  # as the HIL server binds: only a listener blocks it
            try:
                s.bind(("0.0.0.0", HIL_PORT + instance))
            except OSError:
                continue
        return instance
    raise RuntimeError("no free PX4 instance on this machine: 256 are in use")


def _start_px4(cls: Callable, resolved: ResolvedLaunch, instance: int, airframe: str):
    """Start the PX4 SITL peer for this run from ``cls``: the catalog whose pin names the PX4 tree,
    the vehicle's airframe, the run's instance, and a container name and console log of this run's
    own. The name carries the process and the instance, so two runs in one process on two instances
    keep both containers; two on one instance collide on PX4's ports anyway.
    """
    from nexus._src.peers.px4_sitl.runner import PX4_LOG_DIR

    catalog = resolved.tested_config.registry
    os.makedirs(PX4_LOG_DIR, exist_ok=True)
    peer = cls(
        catalog=pathlib.Path(catalog) if catalog else None,
        airframe=airframe,
        instance=instance,
        name=f"nexus-px4-{os.getpid()}-{instance}",
        log_path=os.path.join(PX4_LOG_DIR, f"px4-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.log"),
    )
    peer.start()
    return peer
