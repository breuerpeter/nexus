"""The one shared Kit frame service: every RTX sensor rides a single Kit update per host-seam pass.

Isaac Sim contributes **rendering only**: physics is the same ``NewtonPhysics``, the framework's own
ModelBuilder + solvers, that every runtime steps. Kit never simulates anything:
``/app/player/playSimulations`` stays forced off, and nothing authors a ``UsdPhysics.Scene``. The
frame composes the vehicle + scene Universal Scene Description (USD) files onto a Kit stage purely
for RTX, and writes every vehicle body prim's Fabric ``omni:fabric:worldMatrix`` from the
framework's own model ``body_q`` each rendered frame, the exact mechanism Isaac's own Newton backend
used to sync Fabric, unified with the mounted-sensor pose writes. Body prims only; descendant meshes
compose beneath.

:class:`RtxFrame` owns the render-stage composition, vehicle + scene + sky/handler, the
vehicle/sensor pose writes with epoch pairing, the pre-lockstep warm/drain, and the render-loop
configuration: sync rendering, manual loop mode, the ``/ExternalSimulationTime`` sensor clock, the
playing timeline. Physics stays only the captured lockstep loop: the timeline plays only as the RTX
sensor/render clock.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time

from nexus._src.core import logger
from nexus._src.vehicle.sensors.rtx_stage import VEHICLE_ROOT, discover_rtx_prims, pose_matrix


class RtxConfig:
    """Resolution / world / rate / encoder settings for the RTX render. Optics live in the vehicle
    USD: the camera prim carries focalLength/apertures/xformOp:transform. A ``rtx:`` block in the
    scenario config overrides any field.
    """

    def __init__(self, physics_dt: float, overrides: dict | None = None):
        o = dict(overrides or {})
        # Resolution FALLBACK only: the authority is the camera prim's authored sensor:width/height
        # since the vehicle USD owns its sensors; these apply to un-attributed legacy assets, and an
        # explicit rtx: override in the scenario config still wins for experiments.
        self.width = int(o.get("width", 1280))
        self.height = int(o.get("height", 720))
        # The resolved scene USD path; "" = no scene. Generic scenes compose on the physics
        # stage; the frame only needs the path to find a scene HANDLER in nexus._src.scene,
        # for example the cesium globe, whose streamed content no stage reference can carry.
        self.world = str(o.get("world", ""))
        # Geodetic anchor for a streamed world, {lat, lon}: from the registry scene's
        # geodetic_origin; the cesium handler falls back to the scenario Global Positioning System (GPS)
        # origin so the globe and the GPS sensor agree on where local (0,0,0) is on Earth.
        self.georef = o.get("georef")
        # Camera-rate FALLBACK only; the authority is the prim's authored sensor:rate_hz. 24 fps is the
        # production rate: each render window is ~15-18 ms of serial main-thread work, so rate
        # directly sets the Real Time Factor (RTF); 24 holds >=1x realtime with two cameras on the
        # photoreal cesium world.
        self.render_hz = float(o.get("render_hz", 24.0))

        self.rtsp_url = str(o.get("rtsp_url", "rtsp://127.0.0.1:8554/cam1"))
        self.bitrate = str(o.get("bitrate", "6M"))


class RtxFrame:
    """The one shared Kit frame service for all RTX sensors, and the render-stage owner.

    Constructed by the runtime builder inside the booted Kit app, after ``NewtonPhysics`` has built
    the model, since the pose-write table maps stage prims onto model bodies; composes the Kit stage
    itself: vehicle + scene + sky/handler. Handed to the orchestrator as the lifecycle-only
    ``renderer`` seam, ``on_physics_ready``/``close``, while sensors call :meth:`update` per due
    sample.
    """

    def __init__(self, physics, vehicle_builder=None, *, cfg: dict | None = None):
        self._physics = physics
        self.cfg = RtxConfig(physics.sim_dt, (cfg or {}).get("rtx"))
        self._gps_init = (cfg or {}).get("gps", {}).get("init")  # streamed-world georef fallback: the GPS origin
        self._failed = False
        self._closed = False
        self._last_update = None  # wall-clock of the last Kit frame; coalesces N due sensors -> 1 update
        # Synchronous rendering: required for reading frames per update, the Synthetic Data Generation (SDG)
        # rule that annotator reads under async rendering return the earlier frame. The throttling
        # extension would re-enable async whenever the timeline isn't playing; keep its hands off.
        import carb

        s = carb.settings.get_settings()
        s.set("/exts/isaacsim.core.throttling/enable_async", False)
        # Synchronous rendering is load-bearing: epoch pairing rests on the annotator-read
        # contract. Measured under async rendering: no speedup and corrupt frames, racing poses.
        s.set("/app/asyncRendering", False)
        s.set("/app/asyncRenderingLowLatency", False)
        # The lockstep loop paces renders by sim-time decimation; Kit's rate limiter would only add sleeps
        # inside the update call, on the lockstep thread.
        s.set("/app/runLoops/main/rateLimitEnabled", False)
        # Kit never simulates: physics is the framework's own, the one NewtonPhysics every runtime
        # steps, and the Kit stage is render-only. The player must not step any engine against the
        # vehicle USD's authored UsdPhysics prims; forced off once, for the app's lifetime. Nothing
        # authors a UsdPhysics.Scene either.
        s.set("/app/player/playSimulations", False)
        # /rtx/hydra/supportMultiTickRate must stay True: disabling it kills the RTX sensor pipeline.
        self.needs_timeline = False  # set by any mounted RTX sensor: render clock = playing timeline
        self._time_attr = None  # /ExternalSimulationTime(omni:time): the multitick sensor clock this frame drives
        self._tl = None  # the Kit timeline; this frame drives its clock too, since play() alone never advances it
        self._update_app = None  # isaacsim.core.experimental.utils.app.update_app, set at on_physics_ready
        self.mounted: list = []  # every RtxMountedSensor; the frame writes all their poses per render
        self._epoch = None  # (body_q copy, sim_time) of the last fabric sync = what the next render shows
        self.rendered_epoch_time: float | None = None  # sim time of the state the last render displayed
        self.quiet_window_hooks: list = []  # sensor construction deferred to the pre-lockstep quiet window
        self._prof = None  # orchestrator-owned LoopProfiler for detail spans, via set_profiler
        # The Kit threading rule, GH #39: all Kit pumping executes on the thread that built this
        # frame, the thread that booted Kit: the orchestrator loop's own thread on the command-line
        # path, the user script's main thread under `nexus script`. Kit's update path isn't
        # reentrant across threads: two concurrent pumpers deadlock inside omni.kit.app, and a
        # lone off-main pumper wedges. Nothing crosses a thread to get here any more: the sim is
        # step-driven on the thread that booted Kit, GH #70, so this is an assertion, not a queue.
        self._kit_thread = threading.current_thread()
        # The scene type's code hook, claimed by USD content in nexus._src.scene; None for
        # data-only scenes, mesh/splat, referenced onto the render stage at construction, and the bare sky.
        from nexus._src.scene import handler_for

        self._scene = handler_for(self.cfg.world)
        self._benchmark = None  # --benchmark: Isaac benchmark recorders, the system envelope

        # Lazy Kit/replicator imports, only valid once SimulationApp has booted.
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.usd

        self._rep = rep
        self._app = omni.kit.app.get_app()

        # RTX Real-Time 2.0, the platform default since Kit 108: real-time path tracing with
        # Deep Learning Super Sampling (DLSS) Ray Reconstruction. AA/upscaler stays the platform
        # default too: DLSS, aa/op=3, pinned to Performance mode, since Auto tends to pick Quality,
        # which does reach offscreen render products. Flight-verified: frames sharp, ~20% render-time
        # win over FXAA.
        s.set("/rtx/rendermode", "RealTimePathTracing")
        s.set("/rtx/post/dlss/execMode", 0)
        # Single-GPU box: multi-GPU tiling bookkeeping is pure overhead.
        s.set("/rtx/realtime/mgpu/enabled", False)
        # Viewport guides, the infinite grid + axis lines, composite into offscreen render products too,
        # visible across any see-through ground: cesium, where the flat collider's visual stays hidden.
        s.set("/app/viewport/grid/enabled", False)
        s.set("/app/viewport/show/grid", False)

        # --- the render stage IS the scene USD, opened as the root stage ---
        # The scene asset is the single authority on its own world: geometry, LIGHTS, since every
        # scene usdz authors the sky and there is no in-code sky, and stage render settings, the
        # root layer's customLayerData.renderSettings, which Kit auto-applies on open, so no applier
        # code. Purely visual: no UsdPhysics.Scene, no ground collider, since physics owns its
        # own ground plane; authored UsdPhysics prims compose through but nothing ever simulates them.
        import isaacsim.core.experimental.utils.stage as stage_utils
        from pxr import Gf, UsdGeom

        ctx = omni.usd.get_context()
        if self.cfg.world:
            if not ctx.open_stage(self.cfg.world):
                raise RuntimeError(f"could not open the scene USD as the render stage: {self.cfg.world!r}")
        else:
            stage_utils.create_new_stage()  # scene-less dev/probe fallback: an empty, unlit stage
        stage = ctx.get_stage()
        # Every runtime edit lands on the SESSION layer, never dirtying the asset: the vehicle
        # reference, the scene start-translate, a handler's token/georef injection, tile cameras.
        stage.SetEditTarget(stage.GetSessionLayer())
        if vehicle_builder is not None:
            self._compose_vehicle(stage, vehicle_builder)
        if self.cfg.world and self._scene is None:
            # The registry ``start`` places the scene: one -start translate on the asset's single
            # root prim, session-only, while the vehicle composes at the origin. The root's convention:
            # a plain Xform, defaultPrim, no authored xformOps; see scripts/assets/scene_root.py.
            sx, sy, sz = (cfg or {}).get("scene_start") or (0.0, 0.0, 0.0)
            root = stage.GetDefaultPrim()
            if (sx, sy, sz) != (0.0, 0.0, 0.0):
                if root and root.IsValid():
                    UsdGeom.Xformable(root).AddTranslateOp().Set(Gf.Vec3d(-sx, -sy, -sz))
                else:
                    logger.warning("RtxFrame: scene has no defaultPrim root: start placement skipped")
        self._build_world(stage)
        # The vehicle pose-write table: per model body, the matching stage prim's Fabric worldMatrix
        # attr, written from the framework's own body_q each rendered frame; meshes compose beneath
        # the body prims, exactly as Isaac's own Newton-backend Fabric sync wrote them.
        self._veh_writes: list = []  # (usdrt attr, body_index)
        self._Gf = self._rtGf = None  # set with the pose table: Gf module refs for the writes
        if vehicle_builder is not None:
            self._build_pose_table(stage)
        # Loop-runner + Fabric sim-period config must precede render-product creation by the sensors,
        # constructed right after this frame: RenderingManager.set_dt bakes the Fabric sim period
        # into SimStageWithHistory instances at their creation and existing histories have no
        # setter. Configuring it in on_physics_ready, after the sensors built their render
        # products, left the sensor pipeline on the default period, one ingredient of the
        # view-transform latch, GH #32. The dt is the highest AUTHORED camera rate; the vehicle
        # USD is the authority, and the fallback matches the legacy default.
        try:
            from isaacsim.core.rendering_manager import RenderingManager

            rates = []
            for path in discover_rtx_prims(stage)["camera"]:
                prim = stage.GetPrimAtPath(path)
                attr = prim.GetAttribute("sensor:rate_hz")
                if attr and attr.HasAuthoredValue():
                    rates.append(float(attr.Get()))
            self._render_dt = 1.0 / (max(rates) if rates else float(self.cfg.render_hz or 30.0))
            RenderingManager.set_dt(self._render_dt)
            logger.info(f"RtxFrame: render dt {self._render_dt * 1000.0:.1f}ms (set before render products)")
        except Exception as exc:
            logger.warning(f"RtxFrame: RenderingManager unavailable at construction ({exc!r})")
        # Prewarm, the cold shader compile, waits for on_physics_ready(): rendering before the
        # Newton solver kernels compile shuts the Kit app down.
        logger.info(f"RtxFrame: shared Kit frame service up (world={self.cfg.world!r})")

    def _compose_vehicle(self, stage, vehicle_builder) -> None:
        """Reference the vehicle USD at ``/Vehicle``, its own root, never under the scene's root
        prim, which carries the session start-translate; the vehicle composes at the origin. Applies
        the start pose from ``USDBuilder.spawn_pose``, the shared placement seam, the same pos +
        Forward Right Down (FRD) flip the physics build uses, so the USD-composed poses are right
        before the first Fabric write: camera seeding, cesium tile selection.
        """
        from pxr import Gf, UsdGeom

        usd_path = vehicle_builder.cfg.get("usd_path")
        if not usd_path:
            raise FileNotFoundError("RtxFrame requires a USDBuilder with cfg['usd_path']")
        spawn_pos, att = vehicle_builder.spawn_pose()  # att is a wp.quat (x, y, z, w)
        prim = stage.DefinePrim(VEHICLE_ROOT, "Xform")
        prim.GetReferences().AddReference(str(usd_path))
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in spawn_pos)))
        xf.AddOrientOp().Set(Gf.Quatf(float(att[3]), float(att[0]), float(att[1]), float(att[2])))

    def _build_pose_table(self, stage) -> None:
        """Map each model body onto its ``/Vehicle`` stage prim by leaf-name match, the same
        convention :class:`RtxMountedSensor` mounts by, and create its Fabric ``worldMatrix`` attr.
        Body prims are the only vehicle prims written per frame: a Fabric world matrix on the body
        overrides its USD pose and every descendant mesh composes beneath it, the mechanism Isaac's
        Newton backend itself used for its Fabric sync.
        """
        import omni.usd
        import usdrt
        from pxr import Gf, Usd

        self._Gf = Gf
        self._rtGf = usdrt.Gf
        root = stage.GetPrimAtPath(VEHICLE_ROOT)
        if not root:
            return
        labels = [str(lb).rsplit("/", 1)[-1] for lb in self._physics.model.body_label]
        rt_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
        found: dict[str, str] = {}
        for prim in Usd.PrimRange(root):
            leaf = prim.GetName()
            if leaf in labels and leaf not in found:
                found[leaf] = str(prim.GetPath())
                rt_prim = rt_stage.GetPrimAtPath(str(prim.GetPath()))
                attr = rt_prim.CreateAttribute("omni:fabric:worldMatrix", usdrt.Sdf.ValueTypeNames.Matrix4d, True)
                self._veh_writes.append((attr, labels.index(leaf)))
        missing = [lb for lb in labels if lb not in found]
        if missing:
            # Non-fatal: a model body without a stage prim, for example a builder-only body, just isn't
            # rendered; the vehicle's authored bodies must all match or the render shows a corpse.
            logger.warning(f"RtxFrame: no stage prim for model bodies {missing} (leaf-name match under {VEHICLE_ROOT})")
        logger.info(f"RtxFrame: vehicle pose writes -> {len(self._veh_writes)} body prim(s)")

    def set_profiler(self, prof) -> None:
        """Orchestrator handoff: detail spans, pose/sync/render, inside the shared frame."""
        self._prof = prof

    def _on_kit_thread(self, fn):
        """Run ``fn`` here, asserting the caller is on the Kit thread: the affinity guard, GH #39.

        Every Kit call must run on the thread that built this frame. Since the sim is step-driven
        on that thread, GH #70, nothing can reach here from anywhere else, and this is what keeps
        it that way: a future off-thread caller fails loudly instead of wedging inside omni.kit.app.
        See the ``_kit_thread`` note in ``__init__``.
        """
        current = threading.current_thread()
        if current is not self._kit_thread:
            raise RuntimeError(
                f"Kit work called from {current.name!r} but Kit was booted on {self._kit_thread.name!r}. "
                "every Kit call must run on the thread that built RtxFrame (GH #39)"
            )
        return fn()

    def update(self, sim_time: float | None = None) -> bool:
        """Render one Kit frame for every attached render product; all RTX sensors share it.

        Coalesced: N due sensors in the same host-seam pass trigger at most one Kit update. Owns
        epoch pairing for all mounted sensors: under the playing timeline, sensor view transforms and
        mesh fabric state both latch one frame late, so write every sensor's worldMatrix from the
        CURRENT pose and both display state(k-1) at frame k. Pairing must live here, centrally:
        with two render cadences, camera + lidar, per-sensor pairing lets a sensor pose pair against
        another sensor's mesh epoch and body-fixed geometry swims in the image. Also advances both
        render clocks with the lockstep ``sim_time``: the ``/ExternalSimulationTime`` sensor clock
        and the Kit timeline clock, which playback otherwise paces off the render frame count, so
        authored USD time-samples animate at the wrong rate. Returns False when off. Executes
        on the Kit thread, :meth:`_on_kit_thread`.
        """
        return self._on_kit_thread(lambda: self._update_impl(sim_time))

    def _update_impl(self, sim_time: float | None) -> bool:
        if self._failed or self._closed:
            return False
        # Coalesce by epoch: one sim_time = one Kit frame, shared by every sensor due that pass.
        # Wall-clock coalescing is wrong here: the first sensor's render takes longer than any
        # wall window, so siblings re-rendered the same epoch: 2x the render bill, measured.
        if sim_time is not None and sim_time == self._last_update:
            return True  # a sibling sensor already rendered this epoch; reuse the frame
        if sim_time is None:  # wall-clock fallback path: no lockstep time available
            now = time.time()
            if isinstance(self._last_update, float) and (now - self._last_update) < 1e-3:
                return True
            sim_time_key = now
        else:
            sim_time_key = sim_time
        self._last_update = sim_time_key
        try:
            if self.needs_timeline and self._time_attr is not None and sim_time is not None:
                # The sensor clock must advance monotonically and contiguously; a rewind/reset feature
                # would need multiTickRateReset() here.
                self._time_attr.Set(float(sim_time))
                if self._tl is not None:
                    # Left alone, Kit's playback advances this clock by one RENDER frame per app
                    # update, so it counts renders instead of following sim time. Measured: 1/24 s
                    # per render against a 1/30 s sim step; scene animation ran 25% fast and drifted
                    # without bound. Authored USD time-samples resolve against it, so drive it from
                    # the same lockstep sim time as the sensor clock: animation is then exactly
                    # sim-time-locked, and an rrd frame at sim time t shows the scene at sim time t.
                    self._tl.set_current_time(float(sim_time))
            span = self._prof.span if self._prof is not None else contextlib.nullcontext
            with span("pose.read"):
                bq = self._physics.current_state.body_q.numpy().copy()  # host copy: a few bodies
            epoch_bq, epoch_t = self._epoch if self._epoch is not None else (bq, sim_time)
            with span("pose.write"):
                # One write site for everything the render shows, so it all latches together: the
                # vehicle body prims, with meshes composing beneath, and the mounted sensors, all from
                # the CURRENT state. Under the playing timeline both display one frame late, so what
                # this render shows is the earlier write, the epoch pairing.
                for attr, bi in self._veh_writes:
                    w = pose_matrix(self._Gf, bq[bi])
                    attr.Set(self._rtGf.Matrix4d(*(w[i][j] for i in range(4) for j in range(4))))
                for s in self.mounted:
                    s._write_world_pose(bq[s._body_index])  # latches one frame late, with the body prims
                    s._displayed_world = s._compose_world(epoch_bq[s._body_index])  # what this render shows
            self.rendered_epoch_time = epoch_t
            if self._scene is not None:
                with span("scene.follow"):
                    self._scene.follow(self, bq)  # for example the cesium level-of-detail camera follow
            self._epoch = (bq, sim_time)
            if self._prof is not None:
                with self._prof.span("render.kit"):
                    self._render_once()
            else:
                self._render_once()
            return True
        except Exception as exc:
            self._failed = True
            logger.warning(f"RtxFrame disabled after render error (flight continues): {exc!r}")
            return False

    def close(self) -> None:
        self._on_kit_thread(self._close_impl)

    def _close_impl(self) -> None:
        if self._benchmark is not None:
            out = os.path.expanduser(f"~/.cache/nexus/logs/benchmark-{int(time.time())}.json")
            self._benchmark.finish(out)
            self._benchmark = None
        self._closed = True

    # -- internals ---------------------------------------------------------------------
    def _build_world(self, stage) -> None:
        """The scene USD, opened as the root stage, already IS the world: geometry, sky, render
        settings all authored in the asset. A HANDLER-claimed scene, the cesium globe, whose
        streamed content no stage prims can carry, runs its live machinery here; the render-loop
        housekeeping, the default-viewport freeze, applies either way.
        """
        from pxr import UsdGeom

        mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
        if mpu != 1.0:
            # The scene USD IS the root stage, so its units are the render's units, and every pose
            # write, body_q meters as raw stage units, silently scales by 1/mpu against unit-aware
            # scene content; cesium places the globe in stage units. A cm stage, 0.01, USD's
            # fallback for UNAUTHORED metersPerUnit, rendered all translation at one hundredth: GH #32's
            # "world doesn't move"; rotation is unit-free, so it kept working. The render needs meters;
            # re-author the scene, since scripts/assets/ authoring sets metersPerUnit 1.0 explicitly.
            logger.warning(
                f"RtxFrame: scene stage metersPerUnit={mpu}: the render requires METERS (1.0). "
                f"All rendered translation will be scaled by {1.0 / mpu:.0f}x wrong; re-author the "
                "scene USD with metersPerUnit=1 (GH #32)."
            )
        if self._scene is not None:
            self._scene.compose(self, stage)
        if not (self._scene is not None and self._scene.keeps_default_viewport):
            # The boot's default 'Viewport' window renders offscreen every Kit frame for no consumer,
            # since the sensors have their own render products, so freeze it: the perf handbook's
            # headless disable_viewport_updates. A handler can reuse it instead; cesium: tile selection.
            try:
                from omni.kit.viewport.utility import get_active_viewport

                get_active_viewport().updates_enabled = False
                logger.info("RtxFrame: default viewport frozen (no consumer)")
            except Exception:
                pass

    def _setup_nurec_splat(self) -> None:
        """If the First Person View (FPV) world has a Gaussian splat, ``ParticleField3DGaussianSplat``,
        turn on the NuRec splat render path: enable ``omni.rtx.spg`` + run ``nurec_utils.setup_for_rendering``,
        which classifies the stage and applies the ``omni.rtx.spg`` carb overrides. Without this the splat
        records black. Hard prerequisites, validated on Isaac Sim 6.0.1: the stage must reference the splat
        USD so its ``ParticleField3DGaussianSplat`` type composes onto a stage prim; the physics stage compose
        references it onto a TYPELESS child. ``omni.rtx.spg`` must be on, and ``setup_for_rendering`` must
        run on the composed stage. No-op on builds without the splat renderer, since Isaac < 6.0.1 lacks
        ``omni.rtx.spg`` / ``isaacsim.replicator.nurec_utils``: the import/enable fails and the splat just
        stays black, as before. Fault-isolated.
        """
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if not any("ParticleField" in str(p.GetTypeName()) for p in stage.Traverse()):
                return  # mesh world: nothing to set up
            import omni.kit.app

            em = omni.kit.app.get_app().get_extension_manager()
            em.set_extension_enabled_immediate("isaacsim.replicator.nurec_utils", True)
            em.set_extension_enabled_immediate("omni.rtx.spg", True)
            for _ in range(3):
                self._app.update()
            from isaacsim.replicator.nurec_utils.rendering_setup import setup_for_rendering

            ok, nurec, spg, problems = setup_for_rendering(stage)
            logger.info(f"RtxFrame: NuRec splat render path ok={ok} nurec={nurec} spg={spg} problems={problems}")
        except Exception as exc:
            logger.warning(f"RtxFrame: Gaussian-splat render path unavailable (needs Isaac >= 6.0.1): {exc!r}")

    def _render_once(self) -> None:
        """One render step, the experimental-API way: ``isaacsim.core.experimental.utils.app.update_app``,
        documented to process every sensor tick under multitick rendering. No
        play-guard needed: ``/app/player/playSimulations`` stays forced off for the app's lifetime at
        construction; Kit never simulates, physics is the framework's own. Falls back to a raw app
        update if the experimental utils are unavailable.
        """
        if self._update_app is None:
            self._app.update()
        else:
            self._update_app(steps=1)

    def on_physics_ready(self) -> None:
        """Configure + warm the render pipeline in the safe pre-lockstep window; the orchestrator
        calls this after ``physics.reset()`` and BEFORE PX4 lockstep. Drain Kit's deferred extension
        loads, since pumped in the steady loop they stall the PX4 link, warm the RTX pipeline/shaders,
        since rendering before the first solve shuts the app down, then set up the sensor clock and the
        render-loop config. Physics stays untouched: the stage is render-only, so Kit's render setup
        dirtying it invalidates nothing; the old rebuild→pin→resettle dance died with the Isaac
        physics backend. Fault-isolated. Executes on the Kit thread, :meth:`_on_kit_thread`.
        """
        self._on_kit_thread(self._on_physics_ready_impl)

    def _on_physics_ready_impl(self) -> None:
        if self._failed or self._closed:
            return
        t0 = time.time()
        try:
            cheap = 0
            for _ in range(200):  # step 1: drain deferred loads, ROS2 thrash, until app.update settles cheap
                u0 = time.time()
                self._app.update()
                cheap = cheap + 1 if (time.time() - u0) * 1000.0 < 30.0 else 0
                if cheap >= 5:
                    break
            self._setup_nurec_splat()  # turn on the splat render path now that the world has composed
            for _ in range(8):  # step 2: warm the RTX pipeline, shader/denoiser compile; the sensors' own
                self._rep.orchestrator.step()  # sample() skips cold frames, so warming needs no annotator
            if self.needs_timeline:
                # The RTX sensor/render clock; all three parts are the framework's own, Kit never simulates:
                # 1. the timeline plays forever, since RTX annotators collect only during playback, with
                #    /app/player/playSimulations forced off at construction so no engine steps;
                # 2. this frame writes the /ExternalSimulationTime fabric prim, the multitick scheduler's
                #    per-sensor tick clock, with the LOCKSTEP sim time each render: the contract the
                #    physics engine's step callback normally fulfills;
                # 3. this frame sets the timeline's own clock from that same sim time each render. play()
                #    starts playback, which then paces the clock off the render frame count rather
                #    than sim time; USD time-samples resolve against it; see _update_impl.
                import omni.timeline
                import omni.usd
                import usdrt

                tl = omni.timeline.get_timeline_interface()
                tl.set_end_time(1.0e9)  # the default range ends after seconds, silently stopping the clock
                tl.set_looping(False)
                tl.play()
                fabric = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
                tprim = fabric.GetPrimAtPath("/ExternalSimulationTime")
                if not tprim:
                    tprim = fabric.DefinePrim("/ExternalSimulationTime", "")
                tattr = tprim.GetAttribute("omni:time")
                if not tattr:
                    tattr = tprim.CreateAttribute("omni:time", usdrt.Sdf.ValueTypeNames.Double, True)
                self._time_attr = tattr
                self._tl = tl
                logger.info(
                    "RtxFrame: timeline playing; /ExternalSimulationTime AND the timeline clock "
                    "driven by the lockstep sim time"
                )
            # Loop-runner + Fabric sim-period config moved to construction: it must precede
            # render-product creation; see __init__.
            try:
                from isaacsim.core.experimental.utils.app import update_app

                self._update_app = update_app  # the experimental stepper: processes all sensor ticks
            except Exception as exc:
                self._update_app = None
                logger.warning(f"RtxFrame: experimental update_app unavailable ({exc!r}): raw app.update")
            for hook in self.quiet_window_hooks:
                # Build sensor cores in this quiet window: constructing a render product +
                # its CUDA interop while the captured graph replays dies with CUDA error 700.
                hook()
            for _ in range(4):  # let the fresh sensor graph settle before the steady loop
                self._app.update()
            from nexus._src.diagnostics import diagnostics

            if diagnostics.benchmark:  # the shared --benchmark flag
                from .benchmark import KitBenchmark

                self._benchmark = KitBenchmark()
            if self._scene is not None:
                self._scene.on_ready(self)  # for example the cesium streaming drain + ground-align probe
            logger.info(f"RtxFrame: pre-warm + drain {round(time.time() - t0, 1)}s")
        except Exception as exc:
            self._failed = True
            logger.warning(f"RtxFrame disabled after pre-warm error (flight continues): {exc!r}")
