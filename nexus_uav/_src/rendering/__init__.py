"""The render link's host end: the RTX sensors render in the Kit peer, fed poses over a socket.

The Kit peer is a required peer. No vehicle names it: a vehicle declares an RTX sensor with a schema
on a camera or lidar prim, the sensor's class requires the peer, and a run with one starts it,
:mod:`nexus_uav._src.peers.kit`.

:mod:`.link` is the renderer the loop drives and the link the RTX sensors ride. :func:`rtx_renderer`
starts the peer for a run whose vehicle declares a sensor that requires it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from nexus_uav._src.config.catalog import catalog_layers
from nexus_uav._src.core import logger
from nexus_uav._src.core.registry import Registry
from nexus_uav._src.core.run_folder import run_folder
from nexus_uav._src.peers.files import run_files
from nexus_uav._src.peers.kit.runner import KitPeer, resolve
from nexus_uav._src.peers.pins import PeerPin
from nexus_uav._src.vehicle.sensors.rtx_sensor import PEER

from .link import KitRenderer

__all__ = ["PEER", "KitRenderer", "RtxConfig", "rtx_renderer"]


class RtxConfig:
    """Rate and world settings for the RTX sensors.

    Each sensor's schema is the authority for its resolution and rate; ``render_hz`` paces a run with
    no camera. A ``rtx:`` block in the scenario config overrides any field.
    """

    def __init__(self, overrides: dict | None = None):
        o = dict(overrides or {})
        # 24 fps is the production rate: it holds >=1x realtime with two cameras on the photoreal
        # cesium world.
        self.render_hz = float(o.get("render_hz", 24.0))


class RtxRendererFactory:
    """The renderer for a started Kit peer, called after the physics build.

    :meth:`link` maps the model's bodies onto the render stage and returns the :class:`KitRenderer`
    the RTX sensors ride, which the loop hands each of them at ``connect``. :meth:`finish` hands that
    renderer the built sensors. Calling the factory does both and builds the sensors between them, for
    an assembly that builds no other sensor from the vehicle's file.
    """

    def __init__(self, peer: KitPeer, *, log_dir: Path, components: Registry | None = None):
        self._peer = peer
        self._log_dir = log_dir
        self._components = components

    def link(self, physics, vehicle_usd) -> KitRenderer:
        """The render link for this run: each model body paired with its prim on the render stage, over
        the vehicle's stage, which an RTX sensor reads its prim from at ``connect``.
        """
        from pxr import Usd

        from nexus_uav._src.vehicle.sensors.rtx_stage import render_path

        stage = Usd.Stage.Open(str(vehicle_usd.cfg["usd_path"]))
        root_path = str(stage.GetDefaultPrim().GetPath())
        # The body prims the render poses: a model body's label is its prim's path. The rest of the
        # vehicle composes beneath them.
        labels = [str(label) for label in physics.model.body_label]
        bodies = [(i, render_path(label, root_path)) for i, label in enumerate(labels) if stage.GetPrimAtPath(label)]
        missing = [label for label in labels if not stage.GetPrimAtPath(label)]
        if missing:
            logger.warning(f"no vehicle prim for model bodies {missing}: they don't render")
        return KitRenderer(self._peer, bodies=bodies, stage=stage)

    def finish(self, renderer: KitRenderer, sensors: list, vehicle_usd, cfg: dict, model) -> None:
        """Hand `renderer` the RTX sensors built over it, and fill its setup message, which anchors the
        streamed world at the origin `model` holds.
        """
        from nexus_uav._src.diagnostics import diagnostics

        rtx = RtxConfig(cfg.get("rtx"))
        renderer.sensors = sensors
        pos, att = vehicle_usd.spawn_pose()
        tick = cfg["physics"]["dt"]
        # A camera's frame period: its rate's, and the tick's when its rate makes every tick due.
        periods = [max(tick, 1.0 / s.rate) if s.rate > 0.0 else tick for s in sensors if s.output != "points"]
        benchmark = self._log_dir / "benchmark.json"
        renderer.setup = {
            "scene": cfg.get("scene_usd_path"),
            "scene_start": list(cfg.get("scene_start") or (0.0, 0.0, 0.0)),
            # The streamed world's anchor is the model's geodetic origin, which the sensors read too, so the
            # globe and the Global Positioning System (GPS) sensor agree on where local (0,0,0) is on Earth.
            "georef": dict(
                zip(("lat", "lon", "alt"), (float(v) for v in model.nexus.geodetic_origin.numpy()[0]), strict=True)
            ),
            "vehicle": str(vehicle_usd.cfg["usd_path"]),
            "spawn": {"pos": [float(v) for v in pos], "quat_xyzw": [float(v) for v in att]},
            # The shortest camera frame period the vehicle's file declares; the file is the authority.
            "render_dt": min(periods) if periods else 1.0 / rtx.render_hz,
            "benchmark": str(benchmark) if diagnostics.benchmark else None,
        }

    def __call__(self, physics, vehicle_usd, cfg: dict):
        from nexus_uav._src.core import SeedTree
        from nexus_uav._src.vehicle.sensors.declared import build_sensors, requires, sensor_specs

        usd = vehicle_usd.cfg["usd_path"]
        renderer = self.link(physics, vehicle_usd)
        sensors = build_sensors(
            [spec for spec in sensor_specs(usd, self._components) if requires(spec, PEER)],
            usd_path=usd,
            model=physics.model,
            seedtree=SeedTree(cfg.get("seed", 42)),
        )
        self.finish(renderer, sensors, vehicle_usd, cfg, physics.model)
        return renderer, sensors

    def close(self) -> None:
        """Stop the peer before the loop took it over: the build failed after the start."""
        self._peer.stop()


def rtx_renderer(
    vehicle_usd,
    cfg: dict,
    *,
    cache_dir=None,
    peer: Callable = KitPeer,
    pin: PeerPin | None = None,
    components: Registry | None = None,
    log_dir: Path | None = None,
) -> RtxRendererFactory | None:
    """Start the Kit render peer when the vehicle declares a sensor that requires it, and return its renderer factory.

    Call it once the run has fetched its assets and before the slow parts of the build, so Kit boots
    meanwhile. A vehicle that declares no such sensor starts no container, whatever prims it holds.

    Args:
        vehicle_usd: The vehicle's ``VehicleUsd``, whose Universal Scene Description (USD) decides.
        cfg: The scenario config, carrying the resolved scene.
        cache_dir: The asset cache the run fetched into; ``None`` for the default.
        peer: The class that starts the Kit peer, or a callable that builds one: :class:`KitPeer`, or
            its fake in a test.
        pin: The image the Kit peer runs, its pin, from :func:`nexus_uav._src.peers.kit.runner.resolve`;
            ``None`` resolves it against the catalog a run in the working folder finds.
        components: The registry that resolves the vehicle's schemas to classes; ``None`` takes the
            default one.
        log_dir: The run's folder, which takes Kit's console and its benchmark metrics; ``None``
            takes a new run folder.

    Returns:
        The factory the assembly calls after the physics build, or ``None`` for a vehicle that
        declares no sensor that requires the peer.

    Raises:
        KitPeerError: The Cesium fetch, the image pull or the container start failed; the message
            names the cause.
        ValueError: The vehicle declares a sensor wrongly; the message names the prim.
    """
    from nexus_uav._src.assets.resolver import default_cache
    from nexus_uav._src.vehicle.sensors.declared import requires, sensor_specs

    usd = vehicle_usd.cfg.get("usd_path")
    specs = [spec for spec in sensor_specs(usd, components) if requires(spec, PEER)] if usd else []
    if not specs:
        return None
    prims = [spec.prim for spec in specs]
    logger.info(f"the vehicle declares RTX sensors {prims}: they render in the Kit container")
    if pin is None:
        pin = resolve(catalog_layers())
    log_dir = log_dir if log_dir is not None else run_folder()
    files = run_files(usd, cfg.get("scene_usd_path"))
    started = peer(files, cache_dir=cache_dir or default_cache(), pin=pin, log_dir=log_dir)
    started.start()
    return RtxRendererFactory(started, log_dir=log_dir, components=components)
