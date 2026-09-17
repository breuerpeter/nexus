"""Launch glue for the Isaac Sim runtime: the one neutral glue plus the renderer injection.

Everything runtime-agnostic, meaning resolution, scenario mapping, scene threading, and control-kind
routing, lives in ``nexus._src.runtimes.launch``; this wrapper injects the two Isaac-specific bits:

* the **renderer factory**: stands up the :class:`~nexus._src.rendering.frame.RtxFrame`,
  which composes the Kit render stage from the same resolved Universal Scene Description (USD) files,
  plus one host-rate RTX camera/lidar sensor per prim AUTHORED in the vehicle USD. All control kinds
  get it: an RTX vehicle renders whether PX4, the policy, a Proportional Integral Derivative (PID)
  controller, or a Model Predictive Control (MPC) controller is flying.
* a **generous preroll**, since the PX4 Software In The Loop (SITL) container needs 20-30 s to
  create+build before PX4 even boots, and the CUDA default: this is the GPU/RTX runtime, the CPU
  MuJoCo path is 1-2 orders slower and stalls the lockstep flight, and the runtime still honors an
  EXPLICIT ``--device cpu``.

Must run inside a booted Kit app: the factory touches ``isaacsim``/``omni``.
"""

from __future__ import annotations

import pathlib

from nexus._src.config import LaunchConfig, Registry
from nexus._src.core import Orchestrator, logger
from nexus._src.runtimes.launch import _scenario_from_launch
from nexus._src.runtimes.launch import build_from_launch as _neutral_build_from_launch


def _renderer_factory(stream: bool = False):
    """The Isaac renderer seam: ``(physics, vehicle_builder, cfg) -> (RtxFrame, RTX sensors)``.

    Rendering stays sensor-driven, with no flags: the frame composes the render stage, vehicle + scene
    + sky/handler, and every authored camera/lidar prim discovered on it becomes a host-rate sensor
    over the one shared Kit frame. ``stream`` routes each camera to its own MediaMTX path: EO
    cameras to cam1, cam2, …, thermal ones, with ``sensor:modality = "ir"``, to ir1, ir2, …; without it,
    frames go to the Rerun recording only. Fault-isolated: a failure warns and flies renderless.
    """

    def factory(physics, vehicle_builder, cfg):
        try:
            import omni.usd

            from nexus._src.rendering.frame import RtxFrame
            from nexus._src.vehicle.sensors.rtx_camera import RtxCameraSensor
            from nexus._src.vehicle.sensors.rtx_lidar import RtxLidarSensor
            from nexus._src.vehicle.sensors.rtx_stage import discover_rtx_prims, prim_modality
            from nexus._src.vehicle.sensors.rtx_thermal import RtxThermalSensor

            rtx = dict(cfg.get("rtx", {}))
            if cfg.get("scene_usd_path"):  # the frame renders the scene and finds a HANDLER, for example cesium
                rtx["world"] = cfg["scene_usd_path"]
            frame = RtxFrame(physics, vehicle_builder, cfg={**cfg, "rtx": rtx})
            stage = omni.usd.get_context().get_stage()
            rtx_prims = discover_rtx_prims(stage)
            stream_base = frame.cfg.rtsp_url.rsplit("/", 1)[0]  # rtsp://host:port
            # The modality read happens at CONSTRUCTION, while discovery still buckets by prim type alone,
            # and the two modalities number their streams separately: adding an IR camera to a vehicle must
            # never shift cam1/cam2 out from under a Ground Control Station (GCS) that already discovered them.
            sensors, eo_i, ir_i = [], 0, 0
            for prim in rtx_prims["camera"]:
                if prim_modality(stage.GetPrimAtPath(prim)) == "ir":
                    ir_i += 1
                    url = f"{stream_base}/ir{ir_i}" if stream else None
                    sensors.append(RtxThermalSensor(frame, prim, stream_url=url))
                else:
                    eo_i += 1
                    url = f"{stream_base}/cam{eo_i}" if stream else None
                    sensors.append(RtxCameraSensor(frame, prim, stream_url=url))
            sensors += [RtxLidarSensor(frame, prim) for prim in rtx_prims["lidar"]]
            return frame, sensors
        except Exception as exc:
            logger.warning(f"RTX renderer disabled (flight continues): {exc!r}")
            return None, []

    return factory


def build_from_launch(
    launch: LaunchConfig,
    *,
    registry: Registry | None = None,
    cache_dir: str | pathlib.Path | None = None,
    cfg: dict | None = None,
    stream: bool = False,
) -> Orchestrator:
    """Resolve *launch* and assemble the loop runner in the Kit app: the neutral glue with the
    Isaac renderer factory injected. Routes every control kind, since they all render.
    """
    # This honors an explicit --device cpu, as everywhere else: force_cpu only arrives when the user
    # chose cpu, since runtime.device defaults to "auto" and that resolves to CUDA on this GPU runtime.
    cfg = cfg if cfg is not None else _scenario_from_launch(launch)
    return _neutral_build_from_launch(
        launch,
        registry=registry,
        cache_dir=cache_dir,
        cfg=cfg,
        renderer_factory=_renderer_factory(stream=stream),
        preroll_timeout=120.0,  # the PX4 SITL container builds before PX4 boots
    )


def run_from_launch(launch: LaunchConfig, **kwargs) -> Orchestrator:
    """Resolve, assemble, and run a launch through ``Orchestrator.run()``. Returns the orchestrator."""
    from nexus._src.runtimes.launch import attach_default_recorder

    orch = build_from_launch(launch, **kwargs)
    attach_default_recorder(orch, launch)  # observation default, as in the Sim front door
    orch.run()
    return orch
