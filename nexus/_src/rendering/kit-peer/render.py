"""The render stage: the scene and the vehicle composed in Kit purely for RTX, posed by the host.

Kit simulates nothing: ``/app/player/playSimulations`` stays off and nothing authors a
``UsdPhysics.Scene``. The stage is the scene Universal Scene Description (USD) file opened as the
root stage, with the vehicle referenced at ``/Vehicle``. Each frame the host sends a world matrix
per prim path, and :meth:`Renderer.frame` writes them into Fabric's ``omni:fabric:worldMatrix``,
renders once and reads back the due sensors.

Epoch pairing: under the playing timeline, sensor view transforms and Fabric mesh state both show
one frame late, so a render shows the poses of the request before it. The peer stamps a reply with the
sim time it shows, and the first reply shows the stage as composed.

This file uses Kit's stable layers, ``omni.usd``, ``usdrt``, ``omni.timeline``, ``omni.kit.app``,
``carb`` and ``omni.replicator.core``, over Isaac's experimental API, so an Isaac upgrade touches as
little as possible here.
"""

from __future__ import annotations

import os
import time
from fractions import Fraction

import numpy as np

VEHICLE_ROOT = "/Vehicle"


def log(msg: str) -> None:
    print(f"[kit-peer] {msg}", flush=True)


def _set_render_dt(stage, timeline, settings, dt: float) -> None:
    """Give the run loop, the timeline and Fabric one render period, before any render product exists.

    Fabric bakes its sim period into each ``SimStageWithHistory`` when it creates the history, and a
    history has no setter, so this runs before the sensors build their render products. A period set
    after them left the sensor pipeline on the default one, one ingredient of the view-transform
    latch of GH #32.
    """
    from pxr import Usd

    hz = 1.0 / dt
    with Usd.EditContext(stage, stage.GetRootLayer()):
        stage.SetTimeCodesPerSecond(hz)
    timeline.set_time_codes_per_second(hz)
    try:
        from omni.kit.loop import _loop

        loop = _loop.acquire_loop_interface()
        loop.set_manual_step_size(dt)
        loop.set_manual_mode(True)
    except Exception as exc:
        log(f"no manual loop runner ({exc!r}): the loop keeps its own step")
    period = Fraction(dt).limit_denominator(1_000_000_000)
    settings.set_int("/app/settings/fabricDefaultSimPeriodNumerator", period.numerator)
    settings.set_int("/app/settings/fabricDefaultSimPeriodDenominator", period.denominator)


class _Color:
    """An electro-optical camera: one render product with the ``rgb`` annotator."""

    def __init__(self, rep, path: str, width: int, height: int):
        self.path = path
        self._rp = rep.create.render_product(path, (width, height))
        self._rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self._rgb.attach(self._rp)

    def grab(self, shown) -> list:
        data = np.asarray(self._rgb.get_data())
        if data.ndim != 3 or data.size == 0:
            return []  # cold pipeline: no frame yet
        return [("color", np.ascontiguousarray(data[:, :, :3]))]


class _RadianceDepth:
    """A thermal camera: the ``PtSelfIllumination`` Arbitrary Output Variable (AOV) plus depth on one product.

    The AOV returns at the Deep Learning Super Sampling (DLSS) internal resolution, half the authored size under the pinned
    Performance mode, and the depth at full size; the host's thermal post reconciles the two. The
    product must exist before the first rendered frame: a Pt-annotated product created after frames
    have rendered stays empty, measured.
    """

    def __init__(self, rep, path: str, width: int, height: int):
        self.path = path
        self._rp = rep.create.render_product(path, (width, height))
        self._ir = rep.AnnotatorRegistry.get_annotator("PtSelfIllumination")
        self._ir.attach(self._rp)
        try:
            self._depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
            self._depth.attach(self._rp)
        except Exception as exc:
            self._depth = None
            log(f"{path}: no depth annotator, sky unmasked: {exc!r}")

    def grab(self, shown) -> list:
        data = np.asarray(self._ir.get_data())
        if data.ndim != 3 or data.size == 0:
            return []
        out = [("radiance", np.ascontiguousarray(data[..., 0], dtype=np.float32))]  # red carries temperature
        if self._depth is not None:
            d = np.asarray(self._depth.get_data())
            if d.size:
                out.append(("depth", np.ascontiguousarray(d.reshape(d.shape[0], d.shape[1]), dtype=np.float32)))
        return out


class _Points:
    """A lidar: ``IsaacExtractRTXSensorPointCloud`` on the ``OmniLidar`` prim, returned in the world frame.

    Full-scan accumulation is renderer-native, ``omni:sensor:Core:accumulateOutputs`` with the scan's
    ``tickRate`` baked at authoring; the scan advances on the ``/ExternalSimulationTime`` clock the
    frame drives from the host's sim time. The product attaches after the timeline starts, in the
    same place in the warm-up as before the peer existed.
    """

    def __init__(self, rep, path: str, width: int, height: int):
        self.path = path
        self._rep = rep
        self._annotator = None
        self._warned = False

    def attach(self) -> None:
        self._rp = self._rep.create.render_product(self.path, (128, 128))
        self._annotator = self._rep.AnnotatorRegistry.get_annotator("IsaacExtractRTXSensorPointCloud")
        self._annotator.attach(self._rp)
        log(f"{self.path}: lidar pipeline attached")

    def grab(self, shown) -> list:
        if self._annotator is None:
            return []
        d = self._annotator.get_data()
        pts = d.get("data") if isinstance(d, dict) else d
        if pts is None:
            if isinstance(d, dict) and not self._warned:
                self._warned = True
                log(f"{self.path}: no 'data' key in the annotator output {list(d.keys())}")
            return []
        pts = np.asarray(pts)
        if pts.size == 0 or pts.size % 3:
            return []  # between scans, or a malformed buffer
        pts = pts.reshape(-1, 3)
        # Sensor frame to world through the annotator's own transform. Raw reshape, no transpose: the
        # Replicator info transform is row-vector, translation in the last row.
        info = d.get("info", {}) if isinstance(d, dict) else {}
        m = np.asarray(info["transform"], dtype=np.float64).reshape(4, 4) if "transform" in info else None
        if m is not None and abs(m[3, 3] - 1.0) <= 1e-3:
            pts = pts @ m[:3, :3] + m[3, :3]
        elif shown is not None:  # this sensor's pose in the shown epoch, the row-vector world matrix
            pts = pts @ shown[:3, :3] + shown[3, :3]
        return [("points", np.ascontiguousarray(pts, dtype=np.float32))]


_OUTPUTS = {"color": _Color, "radiance_depth": _RadianceDepth, "points": _Points}


class Renderer:
    """The render stage and its frame loop, built from the host's setup message."""

    def __init__(self, setup: dict):
        import carb
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from pxr import Gf, UsdGeom

        self._rep = rep
        self._app = omni.kit.app.get_app()
        self._tl = omni.timeline.get_timeline_interface()
        self._benchmark_path = setup.get("benchmark")
        self._benchmark = None
        self._epoch = None  # (sim time, {path: matrix}) of the last request: what the next render shows
        self._attrs: dict = {}  # prim path -> its Fabric worldMatrix attr, None for a path the stage lacks
        self._time_attr = None
        self.update_s = 0.0  # wall time inside app.update, summed; the rest of a frame is poses and reads
        s = self._settings = carb.settings.get_settings()
        # Synchronous rendering: an annotator read under async rendering returns the earlier frame,
        # which breaks the epoch pairing. The throttling extension re-enables async whenever the
        # timeline isn't playing; keep its hands off.
        s.set("/exts/isaacsim.core.throttling/enable_async", False)
        s.set("/app/asyncRendering", False)
        s.set("/app/asyncRenderingLowLatency", False)
        # The host paces renders by sim time; Kit's rate limiter would only add sleeps.
        s.set("/app/runLoops/main/rateLimitEnabled", False)
        # Kit never simulates: the player must not step any engine against the vehicle USD's
        # authored UsdPhysics prims.
        s.set("/app/player/playSimulations", False)
        # RTX Real-Time 2.0 with Deep Learning Super Sampling (DLSS) pinned to Performance mode:
        # Auto tends to pick Quality. Flight-verified sharp, about a 20% render-time win over FXAA.
        s.set("/rtx/rendermode", "RealTimePathTracing")
        s.set("/rtx/post/dlss/execMode", 0)
        s.set("/rtx/realtime/mgpu/enabled", False)  # one GPU: multi-GPU tiling is pure overhead
        # Viewport guides composite into offscreen render products too, visible across any
        # see-through ground such as cesium.
        s.set("/app/viewport/grid/enabled", False)
        s.set("/app/viewport/show/grid", False)

        # The render stage IS the scene USD, opened as the root stage: its geometry, lights and
        # render settings, customLayerData.renderSettings, which Kit applies on open.
        world = setup.get("scene") or ""
        from cesium_globe import CesiumGlobe

        self._scene = CesiumGlobe() if world and CesiumGlobe.matches(world) else None
        ctx = omni.usd.get_context()
        if world:
            if not ctx.open_stage(world):
                raise RuntimeError(f"could not open the scene USD as the render stage: {world!r}")
        else:
            ctx.new_stage()  # no scene: an empty, unlit stage
        stage = self._stage = ctx.get_stage()
        # Every runtime edit lands on the session layer, never dirtying the asset.
        stage.SetEditTarget(stage.GetSessionLayer())
        self._compose_vehicle(stage, setup["vehicle"], setup["spawn"])
        if world and self._scene is None:
            # The registry start places the scene: one -start translate on the asset's single root
            # prim, session-only, while the vehicle composes at the origin; see scripts/assets/scene_root.py.
            sx, sy, sz = setup.get("scene_start") or (0.0, 0.0, 0.0)
            if (sx, sy, sz) != (0.0, 0.0, 0.0):
                root = stage.GetDefaultPrim()
                if root and root.IsValid():
                    UsdGeom.Xformable(root).AddTranslateOp().Set(Gf.Vec3d(-sx, -sy, -sz))
                else:
                    log("scene has no defaultPrim root: start placement skipped")
        cameras = [x["path"] for x in setup["sensors"] if x["output"] != "points"]
        self._build_world(stage, world, setup.get("georef"), cameras)
        _set_render_dt(stage, self._tl, s, float(setup["render_dt"]))
        log(f"render dt {float(setup['render_dt']) * 1000.0:.1f}ms (set before render products)")
        self.sensors = [
            _OUTPUTS[x["output"]](rep, x["path"], int(x["width"]), int(x["height"])) for x in setup["sensors"]
        ]
        log(f"stage up: world={world!r}, {len(self.sensors)} sensor(s)")

    def _compose_vehicle(self, stage, vehicle: str, spawn: dict) -> None:
        """Reference the vehicle USD at ``/Vehicle``, its own root, never under the scene's root
        prim, which carries the start translate. The start pose is the one the host's physics
        build uses, so the composed poses are right before the first Fabric write.
        """
        from pxr import Gf, UsdGeom

        prim = stage.DefinePrim(VEHICLE_ROOT, "Xform")
        prim.GetReferences().AddReference(vehicle)
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in spawn["pos"])))
        x, y, z, w = (float(v) for v in spawn["quat_xyzw"])
        xf.AddOrientOp().Set(Gf.Quatf(w, x, y, z))

    def _build_world(self, stage, world: str, georef, cameras: list[str]) -> None:
        """The scene opened as the root stage already is the world. A Cesium scene runs its live
        machinery here; otherwise the boot's default viewport freezes, since the sensors have their
        own render products and it would render every frame for no consumer.
        """
        from pxr import UsdGeom

        mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
        if mpu != 1.0:
            # The scene is the root stage, so its units are the render's, and every pose write, in
            # meters, scales by 1/mpu against unit-aware content. A cm stage rendered all translation
            # at one hundredth: GH #32's "world doesn't move". Re-author the scene with metersPerUnit=1.
            log(
                f"scene stage metersPerUnit={mpu}: the render requires METERS (1.0). All rendered "
                f"translation will be scaled by {1.0 / mpu:.0f}x wrong; re-author the scene USD (GH #32)."
            )
        if self._scene is not None:
            self._scene.compose(stage, world, georef, cameras)
            return
        try:
            from omni.kit.viewport.utility import get_active_viewport

            get_active_viewport().updates_enabled = False
            log("default viewport frozen (no consumer)")
        except Exception:
            pass

    def _setup_nurec_splat(self) -> None:
        """Turn on the NuRec splat render path when the world holds a Gaussian splat,
        ``ParticleField3DGaussianSplat``: enable ``omni.rtx.spg`` and run ``setup_for_rendering``,
        which classifies the stage and applies the splat renderer's settings. Without it the splat
        records black. Validated on Isaac Sim 6.0.1; a build without the splat renderer keeps the
        splat black and says so.
        """
        try:
            if not any("ParticleField" in str(p.GetTypeName()) for p in self._stage.Traverse()):
                return  # a mesh world: nothing to set up
            em = self._app.get_extension_manager()
            em.set_extension_enabled_immediate("isaacsim.replicator.nurec_utils", True)
            em.set_extension_enabled_immediate("omni.rtx.spg", True)
            for _ in range(3):
                self._app.update()
            from isaacsim.replicator.nurec_utils.rendering_setup import setup_for_rendering

            ok, nurec, spg, problems = setup_for_rendering(self._stage)
            log(f"NuRec splat render path ok={ok} nurec={nurec} spg={spg} problems={problems}")
        except Exception as exc:
            log(f"Gaussian-splat render path unavailable (needs Isaac >= 6.0.1): {exc!r}")

    def warm(self) -> None:
        """Drain Kit's deferred extension loads, warm the RTX pipeline, and start the sensor clock.

        A render before the pipeline is warm pays the cold shader compile inside the flight; the
        deferred loads make the first frames slow and uneven.
        """
        t0 = time.time()
        cheap = 0
        for _ in range(200):  # until app.update settles cheap
            u0 = time.time()
            self._app.update()
            cheap = cheap + 1 if (time.time() - u0) * 1000.0 < 30.0 else 0
            if cheap >= 5:
                break
        self._setup_nurec_splat()
        for _ in range(8):  # shader and denoiser compile; a grab skips cold frames, so this needs no annotator
            self._rep.orchestrator.step()
        # The RTX sensor clock, all parts driven from the host's sim time:
        # 1. the timeline plays forever, since RTX annotators collect only during playback, with
        #    /app/player/playSimulations off so no engine steps;
        # 2. each frame writes the /ExternalSimulationTime Fabric prim, the multitick scheduler's
        #    per-sensor clock, with the host's sim time;
        # 3. each frame sets the timeline's own clock from the same sim time: left alone, playback
        #    advances it by one render frame per update, so authored USD time-samples animated at
        #    the render rate and drifted from sim time without bound, GH #62.
        import omni.usd
        import usdrt

        self._tl.set_end_time(1.0e9)  # the default range ends after seconds, silently stopping the clock
        self._tl.set_looping(False)
        self._tl.play()
        fabric = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
        tprim = fabric.GetPrimAtPath("/ExternalSimulationTime")
        if not tprim:
            tprim = fabric.DefinePrim("/ExternalSimulationTime", "")
        tattr = tprim.GetAttribute("omni:time")
        if not tattr:
            tattr = tprim.CreateAttribute("omni:time", usdrt.Sdf.ValueTypeNames.Double, True)
        self._time_attr = tattr
        self._fabric = fabric
        for sensor in self.sensors:
            if isinstance(sensor, _Points):
                sensor.attach()
        for _ in range(4):  # let the fresh sensor graph settle before the flight
            self._app.update()
        if self._benchmark_path:
            from kit_benchmark import KitBenchmark

            self._benchmark = KitBenchmark()
        if self._scene is not None:
            self._scene.on_ready(self._app)  # the Cesium streaming drain
        log(f"warm-up + drain {time.time() - t0:.1f}s")

    def _attr(self, path: str):
        """The Fabric world-matrix attr of ``path``, created on first sight; ``None`` if the stage lacks the prim.

        A world matrix in Fabric overrides the prim's USD pose, and every descendant mesh composes
        beneath it, so body prims and sensor prims are the only ones the host poses.
        """
        if path not in self._attrs:
            import usdrt

            prim = self._fabric.GetPrimAtPath(path)
            if not prim:
                log(f"no stage prim at {path}: its pose is not rendered")
                self._attrs[path] = None
            else:
                self._attrs[path] = prim.CreateAttribute(
                    "omni:fabric:worldMatrix", usdrt.Sdf.ValueTypeNames.Matrix4d, True
                )
        return self._attrs[path]

    def frame(self, t: float, paths: list[str], mats: np.ndarray, due: list[int]) -> tuple[float, list]:
        """Pose every prim the host sent, render once, and read back the due sensors.

        Returns the sim time the render shows and ``[(sensor index, output name, array), …]``.
        """
        import usdrt
        from pxr import Gf

        self._time_attr.Set(float(t))
        self._tl.set_current_time(float(t))
        poses = dict(zip(paths, mats, strict=True))
        for path, m in poses.items():
            attr = self._attr(path)
            if attr is not None:
                attr.Set(usdrt.Gf.Matrix4d(*(float(v) for v in m.ravel())))
        if self._scene is not None:
            self._scene.follow(
                {
                    x.path: Gf.Matrix4d(*(float(v) for v in poses[x.path].ravel()))
                    for x in self.sensors
                    if x.path in poses
                }
            )
        shown_t, shown = self._epoch if self._epoch is not None else (t, poses)
        self._epoch = (t, poses)
        t0 = time.perf_counter()
        self._app.update()
        self.update_s += time.perf_counter() - t0
        out = []
        for i in due:
            sensor = self.sensors[i]
            out += [(i, name, arr) for name, arr in sensor.grab(shown.get(sensor.path))]
        return shown_t, out

    def close(self) -> None:
        if self._benchmark is not None and self._benchmark_path:
            self._benchmark.finish(os.path.expanduser(self._benchmark_path))
            self._benchmark = None
