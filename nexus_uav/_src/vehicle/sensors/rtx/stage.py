"""The render stage: one ovstage and one ovrtx renderer per run, in the loop's process, which every RTX sensor of the
run requires by the name ``ovrtx``.

The build makes it once for a vehicle that declares an RTX sensor, as it makes a required peer, and its start
populates the stage from the run's scene and vehicle files, the scene shifted by its start as the physics shifts it.
The loop hands it to each RTX sensor at ``connect``, where the sensor asks for a render product of its own. At a
tick where any sensor is due, the first sensor's host stage does the tick's work: a Warp kernel turns ``body_q``
into world matrices, ovstage takes them through DLPack, the renderer steps every product once with the elapsed sim
time, and each due sensor takes its output. So a frame shows the poses of the tick that asked for it.

The stage talks to the renderer through six calls and no other: ``population.open_usd`` and
``add_usd_reference``, ``write_attribute``, ``advance_write_floor``, ``attach_ovstage``, ``step`` and ``map``.
ovrtx and ovstage are the extra ``nexus-uav[rtx]``; this module imports them only when a run starts the stage, so
a run with no RTX sensor loads neither.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import warp as wp

from nexus_uav._src.core import logger

NAME = "ovrtx"
# The vehicle's root on the render stage: its own root prim, never under the scene's root, which carries the
# scene's start shift; the vehicle composes at the world origin.
VEHICLE_ROOT = "/Vehicle"
RENDER_MODE = "RealTimePathTracing"
LIDAR_SCHEMA = "NexusLidarAPI"  # the schema that declares a lidar, on an `OmniLidar` prim
SCAN_SCHEMA = "OmniSensorGenericLidarCoreAPI"  # what the renderer needs on that prim to scan it
EXTRA = "nexus-uav[rtx]"
# The render vars each output reads, by the name a sensor's `output` gives.
_VARS = {
    "color": (("LdrColor", "LdrColor"),),
    "radiance_depth": (("PtSelfIllumination", "PtSelfIllumination"), ("Depth", "DistanceToImagePlaneSD")),
    "points": (("PointCloud", "PointCloud"),),
}


def render_path(path: str, root: str) -> str:
    """A vehicle prim's path on the render stage, where the vehicle's root prim sits at ``/Vehicle``."""
    return VEHICLE_ROOT + str(path)[len(str(root)) :]


@wp.kernel(enable_backward=False)
def body_world_matrices(
    body_q: wp.array(dtype=wp.transform), bodies: wp.array(dtype=int), out: wp.array(dtype=wp.mat44d)
):
    """Each bound body's pose as a row-vector world matrix, as Universal Scene Description (USD) holds one: the
    transpose of the rotation, and the translation in the last row.
    """
    i = wp.tid()
    xf = body_q[bodies[i]]
    p = wp.transform_get_translation(xf)
    r = wp.quat_to_matrix(wp.transform_get_rotation(xf))
    # fmt: off
    out[i] = wp.mat44d(
        wp.float64(r[0, 0]), wp.float64(r[1, 0]), wp.float64(r[2, 0]), wp.float64(0.0),
        wp.float64(r[0, 1]), wp.float64(r[1, 1]), wp.float64(r[2, 1]), wp.float64(0.0),
        wp.float64(r[0, 2]), wp.float64(r[1, 2]), wp.float64(r[2, 2]), wp.float64(0.0),
        wp.float64(p[0]), wp.float64(p[1]), wp.float64(p[2]), wp.float64(1.0),
    )
    # fmt: on


def applied(prim) -> list[str]:
    """The API schemas `prim` applies, by name, those this process holds no definition of included."""
    ops = prim.GetMetadata("apiSchemas")
    return list(ops.GetAddedOrExplicitItems()) if ops is not None else []


def _product(scope: str, sensor: str, output: str, width: int, height: int) -> str:
    """The text of a render product over the prim at `sensor`, with the render vars `output` reads, under `scope`."""
    vars_ = _VARS[output]
    defs = "\n".join(
        f'        def RenderVar "{name}"\n        {{\n            uniform string sourceName = "{source}"\n'
        + ('            token[] channels = ["Coordinates", "Counts"]\n' if output == "points" else "")
        + "        }"
        for name, source in vars_
    )
    ordered = ", ".join(f"</{scope}/Vars/{name}>" for name, _ in vars_)
    camera = (
        ""
        if output == "points"
        else f"""        uint[] deviceIds = [0]
        token omni:rtx:background:source:type = "domeLight"
        token omni:rtx:rendermode = "{RENDER_MODE}"
        bool omni:rtx:rt:ecoMode:enabled = false
        token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]
        uniform int2 resolution = ({width}, {height})
"""
    )
    schemas = "" if output == "points" else ' (prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1"])'
    return f"""#usda 1.0
(defaultPrim = "{scope}")
def Scope "{scope}"
{{
    def RenderProduct "RenderProduct"{schemas}
    {{
        rel camera = [<{sensor}>]
{camera}        rel orderedVars = [{ordered}]
    }}
    def "Vars"
    {{
{defs}
    }}
}}
"""


class RenderStage:
    """One run's render stage: the stage the RTX sensors render on and the renderer that draws it.

    The build makes it with the run's values and starts it before the physics build. Each RTX sensor hands
    itself to :meth:`add` at ``connect``, and calls :meth:`tick` from its host stage.

    Args:
        files: The scene description files the run reads; the stage reads its two roots by name instead.
        vehicle_file: The vehicle's root file, which the stage references at ``/Vehicle``.
        scene_file: The scene's root file, the stage's root layer; ``None`` renders the vehicle alone.
        scene_start: The scene-frame point the run places at the world origin, which shifts the scene; ``None``
            shifts nothing.
        log_dir: The run's folder: the renderer's console goes to ``ovrtx/console.log`` in it. ``None`` leaves it
            where ovrtx puts it, in its own install folder.
        **run: The run's other values, its pin and its asset cache, which the stage needs none of.
    """

    name = NAME
    clock = "on_request"

    def __init__(
        self,
        files: list = (),
        *,
        vehicle_file: str,
        scene_file: str | None = None,
        scene_start=None,
        log_dir=None,
        **run,
    ) -> None:
        self.vehicle_file = str(vehicle_file)
        self.log_path = Path(log_dir) / NAME / "console.log" if log_dir is not None else None
        self.scene_file = str(scene_file) if scene_file else None
        self.scene_start = tuple(float(v) for v in (scene_start or (0.0, 0.0, 0.0)))
        self.vehicle_stage = None  # the vehicle's own stage, which a sensor reads its prim from at connect
        self._ovrtx = self._ovstage = None
        self._stage = self._renderer = self._paths = None
        self._ordinal = 0
        self._sensors: list = []
        self._products: dict = {}  # a sensor's product path, by the sensor
        self._query = self._path_list = None
        self._bodies: wp.array | None = None
        self._mats: wp.array | None = None
        self._last_t: float | None = None
        self._last_step = 0.0
        self._warm = False
        self._scans: dict = {}  # each lidar's points since its last due tick, by the lidar

    # -- the lifecycle the build and the loop drive ---------------------------------------------
    def start(self) -> None:
        """Open the vehicle, check every lidar can scan, create the stage and the renderer, and populate the files.

        Raises:
            ValueError: A lidar's prim lacks the schema the renderer scans with; the message names the prim and the
                schema.
            ImportError: ovrtx or ovstage doesn't import; the message names the extra that brings them.
        """
        from pxr import Usd

        self.vehicle_stage = Usd.Stage.Open(self.vehicle_file)
        for prim in self.vehicle_stage.Traverse():
            names = applied(prim)
            if LIDAR_SCHEMA in names and SCAN_SCHEMA not in names:
                raise ValueError(
                    f"{prim.GetPath()}: the lidar's prim lacks {SCAN_SCHEMA}, which the renderer needs to scan it; "
                    "apply it on the prim"
                )
        os.environ.setdefault("OVRTX_SKIP_USD_CHECK", "1")  # ovrtx links its own USD, and this process holds usd-core
        try:
            import ovrtx
            import ovstage
        except ImportError as exc:
            raise ImportError(f"the RTX sensors render with ovrtx and ovstage; install {EXTRA} ({exc})") from exc
        from ovstage import population

        self._ovrtx, self._ovstage = ovrtx, ovstage
        ovrtx.register_schema_paths()
        self._stage = ovstage.Stage(
            f"nexus.{os.getpid()}",
            config=ovstage.StageConfig(
                runtime_default_hierarchy_computation_model=ovstage.HierarchyComputationModel.GPU_INCREMENTAL
            ),
        )
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._renderer = ovrtx.Renderer(
            ovrtx.RendererConfig(
                log_level="warn",
                log_file_path=str(self.log_path) if self.log_path is not None else None,
                keep_system_alive=True,
                read_gpu_transforms=True,  # halves the step, measured
                motion_bvh=ovrtx.MotionBvh.DISABLE,  # a frame costs 7 ms, not 17 ms, once the props spin, measured
                texture_streaming_mode=ovrtx.TextureStreamingMode.SYNCHRONOUS,
            )
        )
        self._ordinal = 1
        if self.scene_file:
            population.open_usd(self._stage, self.scene_file, ordinal=self._ordinal)
        population.add_usd_reference(self._stage, self.vehicle_file, VEHICLE_ROOT)
        population.apply_usd_changes(self._stage, ordinal=self._ordinal)
        self._stage.advance_write_floor(self._ordinal).wait()
        self._renderer.attach_ovstage(self._stage)
        self._paths = ovstage.PathDictionary(self._stage)
        self._ordinal = 2
        if self.scene_file and any(self.scene_start):
            root = str(Usd.Stage.Open(self.scene_file).GetDefaultPrim().GetPath())
            shift = np.eye(4)
            shift[3, :3] = [-v for v in self.scene_start]
            self._write([root], "omni:resetXformStack", np.ones(1, dtype=np.bool_))
            self._write([root], "omni:xform", self._matrices(shift[None]), semantic=ovstage.AttributeSemantic.MATRIX)
        logger.info(f"render stage up: {self.scene_file} + {self.vehicle_file}")

    def alive(self) -> bool:
        """Whether the renderer is up: from the start until the stop."""
        return self._renderer is not None

    def artifacts(self) -> dict:
        """The renderer's console, under ``ovrtx_log``, once the run has one."""
        return {"ovrtx_log": str(self.log_path)} if self.log_path is not None else {}

    def stop(self) -> None:
        """Release the stage and the renderer. Idempotent, and never raises: it runs in teardown."""
        if self._renderer is None:
            return
        try:
            if self._query is not None:
                self._query.release().wait()
                self._paths.destroy_path_list(self._path_list)
            self._paths.destroy()
            self._renderer.detach_ovstage()
            self._stage.destroy()
        except Exception as exc:
            logger.warning(f"render stage teardown: {exc}")
        finally:
            self._renderer.destroy()
            self._renderer = None

    # -- the sensors' side ----------------------------------------------------------------------
    def add(self, sensor) -> None:
        """Author `sensor`'s render product and its frame rate, and bind the model's bodies on the first sensor.

        `sensor` gives ``path``, its prim's path on this stage, ``name``, ``output``, ``width``, ``height``,
        ``rate``, ``model``, ``prim_path``, ``due(now_ns)`` and ``emit(arrays, t_shown)``.
        """
        ovstage, population = self._ovstage, self._population()
        if self._bodies is None:
            self._bind(sensor.model, str(self.vehicle_stage.GetDefaultPrim().GetPath()))
        scope = f"Render_{sensor.name}"
        population.add_usd_reference_from_string(
            self._stage, _product(scope, sensor.path, sensor.output, sensor.width, sensor.height), f"/{scope}"
        )
        population.apply_usd_changes(self._stage, ordinal=self._ordinal)
        product = f"/{scope}/RenderProduct"
        # A reference drops a relationship that targets a prim outside it, so the stage writes the camera's here.
        self._write(
            [product], "camera", np.array([self._paths.intern_path(sensor.path)], dtype=np.uint64),
            is_array=True, semantic=ovstage.AttributeSemantic.RELATIONSHIP_PATH_ID,
        )  # fmt: skip
        if sensor.output == "points":
            # The renderer completes a scan when its clock passes a period of this rate. A camera authors none: the
            # renderer would hold its frame until a whole period had passed, so the first due tick would get none.
            if sensor.rate > 0.0:
                self._write([sensor.path], "omni:sensor:frameRate", np.array([[float(sensor.rate), 1.0]]))
            # World-frame points, as the lidar has always logged them.
            for attr, value in (("outputFrameOfReference", "WORLD"), ("elementsCoordsType", "CARTESIAN")):
                token = np.array([self._paths.intern_token(value)], dtype=np.uint64)
                self._write(
                    [sensor.path], f"omni:sensor:Core:{attr}", token, semantic=ovstage.AttributeSemantic.TOKEN_ID
                )
            self._scans[sensor] = []
        self._products[sensor] = product
        self._sensors.append(sensor)

    def tick(self, t, state) -> None:
        """At a tick where any sensor is due: pose the bodies, step every product once, hand each due sensor its
        output, stamped with the tick's sim time. A later call at the same sim time returns at once.
        """
        now = float(t.sim_time)
        if now == self._last_t:
            return  # a sibling sensor did this tick's work
        self._last_t = now
        due = [s for s in self._sensors if s.due(t.time_ns)]
        if not due:
            return
        if not self._warm:
            self._warm_up()
        ovstage = self._ovstage
        wp.launch(
            body_world_matrices, dim=len(self._bodies), inputs=(state.body_q, self._bodies), outputs=(self._mats,)
        )
        device = self._mats.device
        self._ordinal += 1
        self._stage.write_attribute(
            self._query, "omni:xform", ordinal=self._ordinal,
            tensors=ovstage.make_dltensor(self._mats, dtype=_matrix_type(ovstage)), is_array=False,
            semantic=int(ovstage.AttributeSemantic.MATRIX),
            cuda_stream=(wp.get_stream(device).cuda_stream or 1) if device.is_cuda else None,
        ).wait()  # fmt: skip
        self._stage.advance_write_floor(self._ordinal).wait()
        products = self._renderer.step(
            render_products=set(self._products.values()), delta_time=max(now - self._last_step, 0.0),
            ordinal=self._ordinal,
        )  # fmt: skip
        self._last_step = now
        for lidar, points in self._scans.items():
            if self._products[lidar] in products:
                points.extend(self._points(products[self._products[lidar]], lidar, state))
        for sensor in due:
            if sensor.output == "points":
                scan = self._scans[sensor]
                self._scans[sensor] = []
                # A lidar gives nothing before its first scan completes, a period after the run's start.
                arrays = {"points": np.concatenate(scan)} if scan else {}
            else:
                product = self._products[sensor]
                if product not in products:
                    continue
                arrays = self._images(products[product], sensor.output, product.rsplit("/", 1)[0])
            if arrays:
                sensor.emit(arrays, now)
        del products  # the step's outputs, which the renderer frees once nothing holds them

    # -- inside ----------------------------------------------------------------------------------
    def _population(self):
        from ovstage import population

        return population

    def _matrices(self, mats: np.ndarray):
        flat = np.ascontiguousarray(mats, dtype=np.float64).reshape(-1)
        return self._ovstage.make_dltensor(flat, dtype=_matrix_type(self._ovstage), shape=[mats.shape[0]], ndim=1)

    def _write(self, paths, attr, tensors, *, is_array=False, semantic=0) -> None:
        listed = self._paths.create_path_list_from_strings(list(paths))
        query = self._stage.query_from_path_list(listed)
        self._stage.write_attribute(
            query, attr, ordinal=self._ordinal, tensors=tensors, is_array=is_array, semantic=int(semantic)
        ).wait()
        query.release().wait()
        self._paths.destroy_path_list(listed)

    def _bind(self, model, root: str) -> None:
        """Pair each model body with its prim on the stage, by label, and pose it as a world matrix from now on. A
        sensor's prim rides its body through the vehicle's own hierarchy, so the bodies are all the stage poses.
        """
        labels = [str(label) for label in model.body_label]
        pairs = [(i, render_path(label, root)) for i, label in enumerate(labels) if label.startswith(root)]
        self._bodies = wp.array([i for i, _ in pairs], dtype=int, device=model.device)
        self._mats = wp.empty(len(pairs), dtype=wp.mat44d, device=model.device)
        self._path_list = self._paths.create_path_list_from_strings([p for _, p in pairs])
        self._query = self._stage.query_from_path_list(self._path_list)
        self._stage.write_attribute(
            self._query, "omni:resetXformStack", ordinal=self._ordinal,
            tensors=np.ones(len(pairs), dtype=np.bool_), is_array=False,
        ).wait()  # fmt: skip

    def _warm_up(self) -> None:
        """Commit the stage and step it three times with no sim time passing: the first steps compile shaders, and
        a frame waits for every texture to load.
        """
        self._warm = True
        self._stage.advance_write_floor(self._ordinal).wait()
        for _ in range(3):
            self._renderer.step(render_products=set(self._products.values()), delta_time=0.0, ordinal=self._ordinal)

    def _images(self, product, output: str, scope: str) -> dict:
        """A camera's or a thermal camera's arrays from its product's newest frame, on the host; `scope` is the
        product's parent, which holds its render vars.
        """
        if not product.frames:
            return {}
        frame = product.frames[-1]
        out = {}
        for name, _ in _VARS[output]:
            var = frame.render_vars.get(f"{scope}/Vars/{name}")
            if var is None:
                continue
            with var.map(device=self._ovrtx.Device.CPU) as mapped:
                out[name] = np.array(np.from_dlpack(mapped))
        if output == "color" and "LdrColor" in out:
            return {"color": np.ascontiguousarray(out["LdrColor"][..., :3])}
        if output == "radiance_depth" and "PtSelfIllumination" in out:
            depth = out.get("Depth")
            return {
                "radiance": out["PtSelfIllumination"][..., 0].astype(np.float32),
                "depth": depth.reshape(depth.shape[0], depth.shape[1]) if depth is not None else None,
            }
        return {}

    def _points(self, product, lidar, state) -> list[np.ndarray]:
        """A lidar's world-frame points from every frame of its product this step, less the rays that hit nothing,
        which the renderer returns at the lidar's own position.
        """
        origin = None
        out = []
        for frame in product.frames:
            for var in frame.render_vars.values():
                with var.map(device=self._ovrtx.Device.CPU) as cloud:
                    n = int(np.from_dlpack(cloud["Counts"])[0])
                    points = np.array(np.from_dlpack(cloud["Coordinates"])[:, :n].T, dtype=np.float32)
                if not n:
                    continue
                if origin is None:
                    origin = _origin(lidar, state)
                out.append(points[np.linalg.norm(points - origin, axis=1) > 1e-3])
        return out


def _origin(sensor, state) -> np.ndarray:
    """`sensor`'s world position at the tick `state` holds: its mount's translation, a row of its row-vector
    mount matrix, turned by its body's orientation and moved to its body's position.
    """
    px, py, pz, x, y, z, w = (float(v) for v in state.body_q.numpy()[sensor.body])
    v = np.asarray(sensor.mount, dtype=np.float64)[3, :3]
    q = np.array([x, y, z])
    turned = v + 2.0 * np.cross(q, np.cross(q, v) + w * v)  # the quaternion (x, y, z, w) turns v
    return (turned + np.array([px, py, pz])).astype(np.float32)


def _matrix_type(ovstage):
    """The ovstage data type that holds one row-vector matrix of doubles: sixteen lanes of 64 bits."""
    return ovstage.DLDataType(code=ovstage.DLDataTypeCode.kDLFloat, bits=64, lanes=16)


__all__ = ["NAME", "VEHICLE_ROOT", "RenderStage", "render_path"]
