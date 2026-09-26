"""The Kit render peer: the RTX sensors render in a container of their own, fed poses over a socket.

A vehicle's Universal Scene Description (USD) file decides: a camera or lidar prim under its root is
an RTX sensor, and a run with one starts the peer.

``kit-peer/`` holds the program that container runs and the Dockerfile of its image: package data
that no host module imports, so no host tier ever loads Kit. :mod:`.peer` builds the image and runs
the container; :mod:`.link` is the renderer seam the loop drives and the link the RTX sensors ride.
:func:`rtx_renderer` starts the peer for a run whose vehicle authors RTX sensor prims.
"""

from __future__ import annotations

import time
from pathlib import Path

from nexus._src.core import logger

from .link import KitRenderer
from .peer import KitPeer, KitPeerError, run_script

__all__ = ["KitPeer", "KitPeerError", "KitRenderer", "RtxConfig", "rtx_renderer", "run_script"]


class RtxConfig:
    """Resolution, rate and stream settings for the RTX sensors.

    The camera prim is the authority for its optics, resolution and rate; these are the fallbacks
    for a prim that authors none. A ``rtx:`` block in the scenario config overrides any field.
    """

    def __init__(self, overrides: dict | None = None):
        o = dict(overrides or {})
        self.width = int(o.get("width", 1280))
        self.height = int(o.get("height", 720))
        # 24 fps is the production rate: it holds >=1x realtime with two cameras on the photoreal
        # cesium world.
        self.render_hz = float(o.get("render_hz", 24.0))
        self.rtsp_url = str(o.get("rtsp_url", "rtsp://127.0.0.1:8554/cam1"))
        self.bitrate = str(o.get("bitrate", "6M"))
        # Geodetic anchor for a streamed world, {lat, lon, alt}: the registry scene's geodetic_origin.
        self.georef = o.get("georef")


class RtxRendererFactory:
    """The renderer seam for a started Kit peer, called after the physics build.

    It maps the model's bodies onto the render stage, builds one RTX sensor per camera and lidar
    prim the vehicle authors, and returns them with the :class:`KitRenderer` they ride.
    """

    def __init__(self, peer: KitPeer, *, stream: bool = False):
        self._peer = peer
        self._stream = stream

    def __call__(self, physics, vehicle_builder, cfg: dict):
        from pxr import Usd

        from nexus._src.diagnostics import diagnostics
        from nexus._src.vehicle.sensors.rtx_camera import RtxCameraSensor
        from nexus._src.vehicle.sensors.rtx_lidar import RtxLidarSensor
        from nexus._src.vehicle.sensors.rtx_stage import discover_rtx_prims, prim_modality, render_path
        from nexus._src.vehicle.sensors.rtx_thermal import RtxThermalSensor

        usd = str(vehicle_builder.cfg["usd_path"])
        stage = Usd.Stage.Open(usd)
        root = stage.GetDefaultPrim()
        root_path = str(root.GetPath())
        rtx = RtxConfig(cfg.get("rtx"))
        # The body prims the render poses: each model body onto the first vehicle prim with its
        # label's leaf name. The rest of the vehicle composes beneath them.
        leaves = [str(label).rsplit("/", 1)[-1] for label in physics.model.body_label]
        bodies, found = [], set()
        for prim in Usd.PrimRange(root):
            leaf = prim.GetName()
            if leaf in leaves and leaf not in found:
                found.add(leaf)
                bodies.append((leaves.index(leaf), render_path(prim.GetPath(), root_path)))
        missing = [leaf for leaf in leaves if leaf not in found]
        if missing:
            logger.warning(f"no vehicle prim for model bodies {missing}: they don't render")

        def body_of(path: str) -> int:
            parent = path.rsplit("/", 2)[-2]  # the mount: the sensor prim's parent, matched by leaf name
            return leaves.index(parent) if parent in leaves else 0

        renderer = KitRenderer(self._peer, bodies=bodies)
        prims = discover_rtx_prims(stage, root_path)
        base = rtx.rtsp_url.rsplit("/", 1)[0]  # rtsp://host:port
        # The factory reads the modality at construction, and the two modalities number their streams apart:
        # adding an IR camera to a vehicle must never shift cam1/cam2 out from under a Ground Control
        # Station (GCS) that already discovered them.
        sensors, eo_i, ir_i = [], 0, 0
        for p in prims["camera"]:
            prim = stage.GetPrimAtPath(p)
            kw = {"path": render_path(p, root_path), "body": body_of(p), "cfg": rtx}
            if prim_modality(prim) == "ir":
                ir_i += 1
                sensors.append(RtxThermalSensor(renderer, prim, stream_url=self._url(base, f"ir{ir_i}"), **kw))
            else:
                eo_i += 1
                sensors.append(RtxCameraSensor(renderer, prim, stream_url=self._url(base, f"cam{eo_i}"), **kw))
        for p in prims["lidar"]:
            sensors.append(
                RtxLidarSensor(renderer, stage.GetPrimAtPath(p), path=render_path(p, root_path), body=body_of(p))
            )
        renderer.sensors = sensors

        pos, att = vehicle_builder.spawn_pose()
        rates = [s.rate_hz for s in sensors if s.output != "points"]
        benchmark = Path.home() / ".cache" / "nexus" / "logs" / f"benchmark-{int(time.time())}.json"
        renderer.setup = {
            "scene": cfg.get("scene_usd_path"),
            "scene_start": list(cfg.get("scene_start") or (0.0, 0.0, 0.0)),
            # The streamed world's anchor falls back to the Global Positioning System (GPS) origin, so the globe and the GPS
            # sensor agree on where local (0,0,0) is on Earth.
            "georef": rtx.georef or cfg.get("sensors", {}).get("gps", {}).get("init"),
            "vehicle": usd,
            "spawn": {"pos": [float(v) for v in pos], "quat_xyzw": [float(v) for v in att]},
            # The highest authored camera rate; the vehicle USD is the authority.
            "render_dt": 1.0 / (max(rates) if rates else rtx.render_hz),
            "benchmark": str(benchmark) if diagnostics.benchmark else None,
        }
        return renderer, sensors

    def _url(self, base: str, name: str) -> str | None:
        return f"{base}/{name}" if self._stream else None

    def close(self) -> None:
        """Stop the peer before the loop took it over: the build failed after the start."""
        self._peer.stop()


def rtx_renderer(vehicle_builder, cfg: dict, *, cache_dir=None, stream: bool = False) -> RtxRendererFactory | None:
    """Start the Kit render peer when the vehicle authors RTX sensor prims, and return its renderer factory.

    Call it once the run has fetched its assets and before the slow parts of the build, so Kit boots
    meanwhile. A vehicle with no camera or lidar prim under its root starts no container.

    Args:
        vehicle_builder: The vehicle's builder, whose USD decides.
        cfg: The scenario config, carrying the resolved scene.
        cache_dir: The asset cache the run fetched into; ``None`` for the default.
        stream: Publish each RTX camera's feed over Real Time Streaming Protocol (RTSP).

    Returns:
        The factory the assembly calls after the physics build, or ``None`` for a vehicle with no
        RTX sensor prims.

    Raises:
        KitPeerError: The image build or the container start failed; the message names the cause.
        ValueError: ``stream`` on a vehicle that authors no camera.
    """
    from nexus._src.assets.resolver import default_cache
    from nexus._src.vehicle.sensors.usd import vehicle_rtx_sensor_prims

    usd = vehicle_builder.cfg.get("usd_path")
    prims = vehicle_rtx_sensor_prims(usd) if usd else []
    if stream and not _cameras(usd, prims):
        raise ValueError("streaming needs a camera in the vehicle USD, and this vehicle authors none")
    if not prims:
        return None
    logger.info(f"the vehicle authors RTX sensors {prims}: they render in the Kit container")
    peer = KitPeer([usd, cfg.get("scene_usd_path")], cache_dir=cache_dir or default_cache())
    peer.start()
    return RtxRendererFactory(peer, stream=stream)


def _cameras(usd, prims: list[str]) -> list[str]:
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd))
    return [p for p in prims if stage.GetPrimAtPath(p).GetTypeName() == "Camera"]
