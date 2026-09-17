"""The base of every RTX sensor: a host-rate sensor mounted on a prim the vehicle USD authored.

The prims are base-body children, discovered from the USD with no flags, and each sensor
self-decimates to its authored physical rate on sim time. The mount rides the vehicle pose: the
frame writes each sensor's world matrix per render, so a frame carries the pose it displays.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from nexus._src.core import logger

from .rtx_stage import pose_matrix

if TYPE_CHECKING:  # the frame is duck-typed at runtime; this is the annotation only
    from nexus._src.rendering.frame import RtxFrame


def _prim_sensor_attr(prim, name: str, fallback, kind: str):
    """Read an authored ``sensor:<name>`` custom attr off an RTX prim; the vehicle USD is the
    single authority for its sensors' render params, resolution and rate. Falls back, with a warning,
    only for legacy assets authored before the attrs existed; re-author with
    ``scripts/assets/author_astro_variants.py`` to silence it.
    """
    attr = prim.GetAttribute(f"sensor:{name}")
    if attr and attr.HasAuthoredValue():
        return attr.Get()
    logger.warning(
        f"{kind} {prim.GetPath()}: no authored sensor:{name}, using the fallback {fallback!r} "
        "(re-author the vehicle USD; the asset is the authority for its sensors)"
    )
    return fallback


class RtxMountedSensor:
    """Shared base for RTX sensors mounted on a vehicle body, camera or lidar: a host-rate ``Sensor``
    over a prim AUTHORED in the vehicle USD as a base-body child.

    The renderable IS a sensor, the framework's one abstraction: discovered from the USD with no flags,
    sampled at the host seam of every loop, self-decimating to ``rate_hz``, never vetoing the captured
    strategy, per ``host_rate``. The code enforces mount rigidity explicitly: fabric worldMatrix has no
    hierarchy, so :meth:`RtxFrame.update` writes each sensor's world matrix,
    W = L_authored * W_body in row-vector Gf, from the same state the body matrices sync from.
    """

    host_rate = True  # host-bound + low-rate: sampled at the host seam, excluded from the capture gate
    KIND = "rtx"

    def __init__(self, frame: RtxFrame, prim_path: str, *, rate_hz: float):
        import omni.usd
        import usdrt
        from pxr import Gf, UsdGeom

        self._frame = frame
        self._logger = None
        self._failed = False
        self.prim_path = prim_path
        self.name = prim_path.rsplit("/", 1)[-1].lower()  # for example 'fpvcam' -> cameras/fpvcam
        self.rate_hz = float(rate_hz)
        self._last_sample = None
        self._last_world = None  # the WRITTEN world matrix, Gf, set per _write_world_pose
        self._displayed_world = None  # the matrix the last render displays; RtxFrame.update sets it
        frame.mounted.append(self)  # the frame writes this sensor's pose per render, the epoch pairing
        frame.needs_timeline = True  # any RTX sensor => the timeline plays, as the sensor/render clock

        usd_stage = omni.usd.get_context().get_stage()
        usd_prim = usd_stage.GetPrimAtPath(prim_path)
        self._local = UsdGeom.Xformable(usd_prim).GetLocalTransformation()  # authored mount, a Gf.Matrix4d
        self._Gf = Gf
        rt_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
        rt_prim = rt_stage.GetPrimAtPath(prim_path)
        self._world_attr = rt_prim.CreateAttribute("omni:fabric:worldMatrix", usdrt.Sdf.ValueTypeNames.Matrix4d, True)
        self._rtGf = usdrt.Gf
        # The body row carrying this sensor: match the parent prim's leaf against the model labels.
        parent_leaf = prim_path.rsplit("/", 2)[-2]
        labels = [
            str(frame._physics.model.body_label[i]).rsplit("/", 1)[-1]
            for i in range(len(frame._physics.model.body_label))
        ]
        self._body_index = labels.index(parent_leaf) if parent_leaf in labels else 0
        logger.info(
            f"{type(self).__name__}: {prim_path} @{self.rate_hz:.0f}Hz -> {self.KIND}/{self.name} "
            f"(mounted on body[{self._body_index}] {parent_leaf!r})"
        )

    def set_logger(self, logger_) -> None:
        """Component-owned logging seam; the orchestrator hands the Logger over, and None = off.

        In debug mode, non-camera sensors log their coordinate frame once, statically, under their USD
        prim path, a child of the body prim, with the fixed local mount transform: the debug scene logs
        a per-tick ``Transform3D`` on each body entity, so the child frame inherits the body pose and
        nothing re-logs per sample. Cameras carry their own Pinhole frustum instead.
        """
        self._logger = logger_
        if logger_ is not None and getattr(logger_, "debug", False) and self.KIND != "camera":
            try:
                loc = self._local  # row-vector Gf: the fixed mount transform relative to the body
                logger_.log_static_frame(
                    self.prim_path,
                    [loc[3][0], loc[3][1], loc[3][2]],
                    [[loc[i][j] for j in range(3)] for i in range(3)],
                    label=self.name,
                )
            except Exception as exc:
                logger.warning(f"{type(self).__name__} {self.name}: static frame logging unavailable: {exc!r}")

    def _compose_world(self, bq):
        """Sensor world matrix, row-vector Gf, for a body pose row: rigid mount, W = L_authored * W_body."""
        return self._local * pose_matrix(self._Gf, bq)  # row-vector: child world = local * parent world

    def _write_world_pose(self, bq) -> None:
        """Set the sensor's Fabric world matrix from a body pose row; epoch pairing: RtxFrame.update."""
        w = self._compose_world(bq)
        self._last_world = w
        self._world_attr.Set(self._rtGf.Matrix4d(*(w[i][j] for i in range(4) for j in range(4))))

    def sample(self, state, env, t, out) -> None:
        """Host-seam sample: due? -> fresh shared Kit frame -> grab -> emit.

        Decimation is on sim time, since ``rate_hz`` is the sensor's physical rate: a 30 Hz camera samples
        30x per simulated second regardless of RTF.
        """
        if self._failed:
            return
        now = float(getattr(t, "sim_time", 0.0)) if t is not None else time.time()
        # Absolute due-time accumulation, not last+period quantized to the tick grid: a 24 Hz camera
        # on 4 ms ticks would land every 44 ms = 22.7 fps; accumulation alternates 40/44 ms = 24.0.
        if self._last_sample is not None and now < self._last_sample:
            return
        period = 1.0 / self.rate_hz
        base = self._last_sample if self._last_sample is not None else now
        self._last_sample = max(base + period, now + 0.5 * period)  # never burst after a stall
        try:
            # The wall-clock fallback, t=None, must never reach the /ExternalSimulationTime clock.
            if not self._frame.update(sim_time=now if t is not None else None):
                return
            t_render = self._frame.rendered_epoch_time  # the epoch the just-rendered frame displays
            if t_render is None:
                t_render = now
            self._grab_and_emit(t_render)  # stamped at the epoch the image actually shows
        except Exception as exc:
            self._failed = True
            logger.warning(f"{type(self).__name__} {self.name} disabled (flight continues): {exc!r}")

    def _grab_and_emit(self, t) -> None:  # per-sensor: annotator grab + emission; t = the frame's sim time
        raise NotImplementedError

    def close(self) -> None:
        pass
