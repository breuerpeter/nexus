"""The Isaac Sim runtime: hosts the fixed-order tick in the Kit app, per architecture.md §12.

The one neutral assembly + launch glue, ``runtimes.assembly`` / ``runtimes.launch``, wires the
same core components, including the same ``NewtonPhysics`` with this repo's ModelBuilder and
solvers, since Isaac's physics backend stays unused and Kit never simulates. This runtime only
injects its renderer factory, see ``.launch``: a Kit stage composed purely for RTX, posed from this
repo's model each rendered frame. The controller still exposes the TCP:4560
Hardware In The Loop (HIL) server PX4 dials into with ``PX4_SIMULATOR=none``; PX4 lockstep still
paces the loop. Sensors, actuator, ``Px4MavlinkController`` and the logging stack run **verbatim**.

Boot order is load-bearing: ``isaacsim``/``omni`` only import once ``SimulationApp`` has started,
so :func:`run_isaacsim` boots the Kit app **first**, then assembles + runs the orchestrator. The
single-package layout means workspace imports are ``nexus._src.<area>``, unambiguous against
NVIDIA Newton's own top-level ``newton_actuators`` in the Kit prebundle, so nothing needs rebinding,
unlike the old 12-package workspace where the bare ``newton_actuators`` name collided.
"""

from __future__ import annotations

import os

# The plain full Isaac experience: with the WORKSPACE physics pins active, NEXUS_PINS_DIR, which
# the container entrypoint installs from uv.lock, boot needs nothing Newton-specific: the same
# newton/warp as the host, no version skew. Without pins, as with stale caches, the bundled builds
# still resolve and everything runs, just one newton minor back.
NEWTON_EXPERIENCE = "/isaac-sim/apps/isaacsim.exp.full.kit"


def apply_pins_pre_boot() -> None:
    """Pre-SimulationApp half of the workspace-pins recipe, a no-op without ``NEXUS_PINS_DIR``.

    Kit's extension importer resolves its bundled ``warp``/``newton`` copies at the sys.meta_path
    FINDER level, so path order alone can't win: the pinned modules must be in ``sys.modules``
    BEFORE the boot. That means warp + the submodules newton lazily pulls, and warp only, because
    newton needs Kit's ``pxr``, see :func:`apply_pins_post_boot`. This also excludes the
    ``isaacsim.pip.newton`` prebundle extension so post-boot newton imports resolve the pins.
    """
    import sys

    pins = os.environ.get("NEXUS_PINS_DIR")
    if not (pins and os.path.isdir(pins)):
        return
    sys.path.insert(0, pins)
    import warp
    import warp.fem
    import warp.optim
    import warp.render  # noqa: F401

    sys.argv += ["--/app/extensions/excluded/0=isaacsim.pip.newton"]


def apply_pins_post_boot() -> None:
    """Post-SimulationApp half: re-front the pins, since the boot inserted extension paths ahead, so
    the now-possible ``import newton``, with Kit's ``pxr`` available, and every later lazy import
    resolve the pinned copies. A no-op without ``NEXUS_PINS_DIR``.
    """
    import sys

    pins = os.environ.get("NEXUS_PINS_DIR")
    if pins and os.path.isdir(pins):
        sys.path.insert(0, pins)
        import warp

        from nexus._src.core import logger

        logger.info(f"workspace physics pins active ({pins}): warp {warp.__version__}")


def run_isaacsim(
    vehicle: str = "astro_max_base",
    *,
    registry: str | None = None,
    control: str = "px4-sitl",
    view: bool = False,
    log: bool = False,
    debug: bool = False,
    stream: bool = False,
    scene: str | None = None,
    geo: str | None = None,
    max_steps: int | None = None,
    rtf: float = 0.0,
) -> None:
    """Boot the Kit app and fly a **registry** vehicle through the core Orchestrator: the same
    ``NewtonPhysics``, RTX-rendered, the Isaac twin of the command-line tool's standalone ``Sim`` path.

    Boots ``SimulationApp`` headless, resolves the launch against the registry, and runs the
    fixed-order tick, serving the PX4 HIL link on TCP:4560 exactly as the standalone runtime does.
    The logging flags carry the framework semantics of ``sim_argparser``: ``view`` serves the Rerun
    recording live on :9876, ``log`` writes the ``.rrd`` to disk, omitting both records nothing, for
    max headless speed; ``debug`` logs the axes-only scene: per-body triads, a small ``.rrd``.
    Rendering is SENSOR-driven, not flagged: the vehicle Universal Scene Description (USD) file's
    authored camera prims become host-rate RTX camera sensors, ``cameras/<name>`` in the recording.
    ``stream`` routes the first camera's feed to NVENC → Real Time Streaming Protocol (RTSP) →
    MediaMTX, per architecture.md §11. ``scene`` overrides the registry default, so a render-only
    First Person View (FPV) world becomes the render world; ``max_steps`` bounds the run.
    """
    from isaacsim import SimulationApp

    # The Isaac runtime is the RTX runtime: boot render-capable ALWAYS. Physics-only vehicles route to
    # the standalone runtime via USD detection, so a cameraless isaacsim run is the exception, and an
    # idle RTX renderer costs nothing per tick: no render products => no Kit frame is ever pumped.
    # multi_gpu must be off: the Gaussian-splat NuRec renderer, omni.rtx.spg, refuses to activate when
    # /renderer/multiGpu/enabled is true, a 6.0.1 validation, → splats render black.
    boot = {"headless": True, "renderer": "RayTracedLighting", "width": 1280, "height": 720, "multi_gpu": False}
    apply_pins_pre_boot()  # workspace newton/warp, see NEWTON_EXPERIENCE; no-op without pins
    sim_app = SimulationApp(boot, experience=NEWTON_EXPERIENCE)
    apply_pins_post_boot()
    try:
        from nexus._src.config import LaunchConfig
        from nexus._src.core import logger

        from .launch import run_from_launch

        logger.info(f"newton-runtime-isaacsim: vehicle={vehicle} control={control} stream={stream}")
        launch = LaunchConfig().set_vehicle(vehicle)  # name / local .usd / None -> the registry default
        launch.registry = registry  # None: the run finds its own catalog
        launch.set_control(control)  # px4-sitl; examples self-assemble under `nexus script`
        if scene is not None:
            launch.set_scene(scene)
        if geo is not None:  # override the scene geodetic origin, for example to fly cesium over any lat/lon
            launch.set_geodetic_origin(*[float(x) for x in geo.split(",")])  # lat,lon[,alt]
        launch.runtime.max_steps = max_steps
        launch.runtime.rtf = rtf  # 0 = unthrottled; 1.0 = wall-clock pacing for human-in-the-loop
        launch.output.view = view  # serve :9876
        launch.output.log = log  # write the .rrd; neither → no recording; Isaac logging is output-only, resilient
        launch.output.debug = debug  # axes-only scene
        run_from_launch(launch, stream=stream)
    finally:
        sim_app.close()


# NOTE, 2026-07-08: this runtime used to turn off omni.anim.behavior.* + isaacsim.ros2.* at boot: the
# former to dodge a render-loop segfault from anim.behavior.core's stale DirectorWindow handler, the
# latter because the ROS2 bridge thrashed startup/shutdown and dropped the PX4 lockstep. On the current
# isaacsim-runtime image both problems have vanished: a full two-camera cesium flight, playing timeline
# + render + PX4 lockstep, takeoff to 50 m, runs with those extensions LOADED and shows no segfault, no
# lockstep drop, and an unchanged Real Time Factor (RTF) of about 1x. Turning them off at runtime was
# itself fragile, since tearing down anim.behavior cascaded into .ui's buggy teardown and C++-aborted
# the process, so the change was a net negative. Removed. If a future image regresses, `git log` this
# file for the guard; the durable fix would be a persistent-autoload opt-out, /persistent/app/exts/enabled,
# so they never load.
