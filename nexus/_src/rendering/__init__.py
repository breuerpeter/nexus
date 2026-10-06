"""The render link's host end: the RTX sensors render in the Kit peer, fed poses over a socket.

The Kit peer is a required peer. No vehicle names it: a vehicle declares an RTX sensor with a schema
on a camera or lidar prim, the sensor's class requires the peer, and a run with one starts it,
:mod:`nexus._src.peers.kit`.

:mod:`.link` is the renderer the loop drives and the link the RTX sensors ride. :func:`rtx_renderer`
starts the peer for a run whose vehicle declares a sensor that requires it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from nexus._src.core import logger
from nexus._src.core.registry import ComponentRegistry
from nexus._src.peers.kit.runner import KitPeer

from .link import KitRenderer

__all__ = ["KitRenderer", "RtxConfig", "rtx_renderer"]

PEER = "kit"  # the name an RTX sensor's class requires the Kit peer by


class RtxConfig:
    """Rate and stream settings for the RTX sensors.

    Each sensor's schema is the authority for its resolution and rate; ``render_hz`` paces a run with
    no camera. A ``rtx:`` block in the scenario config overrides any field.
    """

    def __init__(self, overrides: dict | None = None):
        o = dict(overrides or {})
        # 24 fps is the production rate: it holds >=1x realtime with two cameras on the photoreal
        # cesium world.
        self.render_hz = float(o.get("render_hz", 24.0))
        self.rtsp_url = str(o.get("rtsp_url", "rtsp://127.0.0.1:8554/cam1"))
        self.bitrate = str(o.get("bitrate", "6M"))
        # Geodetic anchor for a streamed world, {lat, lon, alt}: the registry scene's geodetic_origin.
        self.georef = o.get("georef")


class Streams:
    """Where each camera of a run publishes its feed, for a run that streams.

    Each kind of camera numbers its streams apart, ``cam1``, ``cam2`` and ``ir1``: adding an infrared
    camera to a vehicle must never shift ``cam1`` out from under a Ground Control Station (GCS) that
    already discovered it.
    """

    def __init__(self, rtx: RtxConfig, *, enabled: bool):
        self.bitrate = rtx.bitrate
        self._base = rtx.rtsp_url.rsplit("/", 1)[0] if enabled else None  # rtsp://host:port
        self._count: dict[str, int] = {}

    def url(self, kind: str) -> str | None:
        """The next stream address for a camera of `kind`, ``cam`` or ``ir``; ``None`` when the run doesn't stream."""
        if self._base is None:
            return None
        self._count[kind] = self._count.get(kind, 0) + 1
        return f"{self._base}/{kind}{self._count[kind]}"


class RtxRendererFactory:
    """The renderer seam for a started Kit peer, called after the physics build.

    :meth:`link` maps the model's bodies onto the render stage and returns the :class:`KitRenderer`
    the RTX sensors ride. :meth:`finish` hands that renderer the built sensors. Calling the factory
    does both and builds the sensors between them, for an assembly that builds no other sensor from
    the vehicle's file.
    """

    def __init__(self, peer: KitPeer, *, stream: bool = False, components: ComponentRegistry | None = None):
        self._peer = peer
        self._stream = stream
        self._components = components

    def link(self, physics, vehicle_builder, cfg: dict) -> KitRenderer:
        """The render link for this run: each model body paired with its prim on the render stage."""
        from pxr import Usd

        from nexus._src.vehicle.sensors.rtx_stage import render_path

        stage = Usd.Stage.Open(str(vehicle_builder.cfg["usd_path"]))
        root_path = str(stage.GetDefaultPrim().GetPath())
        # The body prims the render poses: a model body's label is its prim's path. The rest of the
        # vehicle composes beneath them.
        labels = [str(label) for label in physics.model.body_label]
        bodies = [(i, render_path(label, root_path)) for i, label in enumerate(labels) if stage.GetPrimAtPath(label)]
        missing = [label for label in labels if not stage.GetPrimAtPath(label)]
        if missing:
            logger.warning(f"no vehicle prim for model bodies {missing}: they don't render")
        renderer = KitRenderer(self._peer, bodies=bodies)
        renderer.streams = Streams(RtxConfig(cfg.get("rtx")), enabled=self._stream)
        return renderer

    def finish(self, renderer: KitRenderer, sensors: list, vehicle_builder, cfg: dict) -> None:
        """Hand `renderer` the RTX sensors built over it, and fill its setup message."""
        from nexus._src.diagnostics import diagnostics

        rtx = RtxConfig(cfg.get("rtx"))
        renderer.sensors = sensors
        pos, att = vehicle_builder.spawn_pose()
        rates = [s.rate for s in sensors if s.output != "points"]
        benchmark = Path.home() / ".cache" / "nexus" / "logs" / f"benchmark-{int(time.time())}.json"
        renderer.setup = {
            "scene": cfg.get("scene_usd_path"),
            "scene_start": list(cfg.get("scene_start") or (0.0, 0.0, 0.0)),
            # The streamed world's anchor falls back to the Global Positioning System (GPS) origin, so the globe and the GPS
            # sensor agree on where local (0,0,0) is on Earth.
            "georef": rtx.georef or cfg.get("sensors", {}).get("gps", {}).get("init"),
            "vehicle": str(vehicle_builder.cfg["usd_path"]),
            "spawn": {"pos": [float(v) for v in pos], "quat_xyzw": [float(v) for v in att]},
            # The highest declared camera rate; the vehicle's file is the authority.
            "render_dt": 1.0 / (max(rates) if rates else rtx.render_hz),
            "benchmark": str(benchmark) if diagnostics.benchmark else None,
        }

    def __call__(self, physics, vehicle_builder, cfg: dict):
        from nexus._src.core import SeedTree
        from nexus._src.vehicle.sensors.declared import build_sensors, requires, sensor_specs

        usd = vehicle_builder.cfg["usd_path"]
        renderer = self.link(physics, vehicle_builder, cfg)
        sensors = build_sensors(
            [spec for spec in sensor_specs(usd, self._components) if requires(spec, PEER)],
            usd_path=usd,
            model=physics.model,
            seedtree=SeedTree(cfg.get("seed", 42)),
            dt=cfg["physics"]["dt"],
            site=None,
            link=renderer,
        )
        self.finish(renderer, sensors, vehicle_builder, cfg)
        return renderer, sensors

    def close(self) -> None:
        """Stop the peer before the loop took it over: the build failed after the start."""
        self._peer.stop()


def rtx_renderer(
    vehicle_builder,
    cfg: dict,
    *,
    cache_dir=None,
    stream: bool = False,
    peer: Callable = KitPeer,
    components: ComponentRegistry | None = None,
) -> RtxRendererFactory | None:
    """Start the Kit render peer when the vehicle declares a sensor that requires it, and return its renderer factory.

    Call it once the run has fetched its assets and before the slow parts of the build, so Kit boots
    meanwhile. A vehicle that declares no such sensor starts no container, whatever prims it holds.

    Args:
        vehicle_builder: The vehicle's builder, whose USD decides.
        cfg: The scenario config, carrying the resolved scene.
        cache_dir: The asset cache the run fetched into; ``None`` for the default.
        stream: Publish each RTX camera's feed over Real Time Streaming Protocol (RTSP).
        peer: The class that starts the Kit peer, or a callable that builds one: :class:`KitPeer`, or
            its fake in a test.
        components: The registry that resolves the vehicle's schemas to classes; ``None`` takes the
            default one.

    Returns:
        The factory the assembly calls after the physics build, or ``None`` for a vehicle that
        declares no sensor that requires the peer.

    Raises:
        KitPeerError: The Cesium fetch, the image pull or the container start failed; the message
            names the cause.
        ValueError: ``stream`` on a vehicle that declares no camera, or a vehicle that declares a
            sensor wrongly; the message names the prim.
    """
    from nexus._src.assets.resolver import default_cache
    from nexus._src.vehicle.sensors.declared import requires, sensor_specs

    usd = vehicle_builder.cfg.get("usd_path")
    specs = [spec for spec in sensor_specs(usd, components) if requires(spec, PEER)] if usd else []
    if stream and not [spec for spec in specs if getattr(spec.cls, "KIND", "") == "cameras"]:
        raise ValueError("streaming needs a camera in the vehicle USD, and this vehicle declares none")
    if not specs:
        return None
    prims = [spec.prim for spec in specs]
    logger.info(f"the vehicle declares RTX sensors {prims}: they render in the Kit container")
    # A run with an override layer opens its vehicle through a root file that stacks the layer on the
    # asset, so the peer reads the files that root file names too.
    from pxr import Sdf

    root = Sdf.Layer.FindOrOpen(str(usd))
    stacked = list(root.subLayerPaths)
    started = peer([usd, *stacked, cfg.get("scene_usd_path")], cache_dir=cache_dir or default_cache())
    started.start()
    return RtxRendererFactory(started, stream=stream, components=components)
