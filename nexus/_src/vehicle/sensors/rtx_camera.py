"""The RTX camera: the electro-optical (EO) feed over a ``Camera`` prim the vehicle USD authored.

One render product plus an ``LdrColor`` annotator, logged as the First Person View image and, when
the launch asks for a stream, pushed to the RTSP publisher.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np

from nexus._src.core import logger

from .rtsp import RtspPublisher
from .rtx_sensor import RtxMountedSensor, _prim_sensor_attr

if TYPE_CHECKING:  # the frame is duck-typed at runtime; this is the annotation only
    from nexus._src.rendering.frame import RtxFrame


class RtxCameraSensor(RtxMountedSensor):
    """RTX camera sensor: rgb annotator on the authored camera prim -> ``Logger.log_image``,
    at ``cameras/<name>``, and, when streaming, the :class:`RtspPublisher`.
    """

    KIND = "cameras"

    def __init__(self, frame: RtxFrame, prim_path: str, *, stream_url: str | None = None):
        import omni.replicator.core as rep
        import omni.usd

        # Render params from the AUTHORED prim, sensor:width/height/rate_hz; the vehicle USD is
        # the single authority, and RtxConfig only covers legacy assets / explicit rtx: overrides.
        cam_prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
        self.width = int(_prim_sensor_attr(cam_prim, "width", frame.cfg.width, "RtxCameraSensor"))
        self.height = int(_prim_sensor_attr(cam_prim, "height", frame.cfg.height, "RtxCameraSensor"))
        rate_hz = float(_prim_sensor_attr(cam_prim, "rate_hz", frame.cfg.render_hz, "RtxCameraSensor"))
        # The --stream consumer is per camera at the CAMERA's resolution/rate.
        self._publisher = (
            RtspPublisher(
                width=self.width,
                height=self.height,
                fps=max(1, round(rate_hz)),
                bitrate=frame.cfg.bitrate,
                rtsp_url=stream_url,
            )
            if stream_url
            else None
        )
        self._rp = rep.create.render_product(prim_path, (self.width, self.height))
        self._rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self._rgb.attach(self._rp)
        # authored intrinsics -> the Rerun Pinhole for the frustum / field-of-view visualization; logged once on set_logger
        self._focal_mm = float(cam_prim.GetAttribute("focalLength").Get() or 12.0)
        self._h_aperture_mm = float(cam_prim.GetAttribute("horizontalAperture").Get() or 36.0)
        self._v_aperture_mm = float(cam_prim.GetAttribute("verticalAperture").Get() or 0.0) or None
        super().__init__(frame, prim_path, rate_hz=rate_hz)

    def set_logger(self, logger_) -> None:
        super().set_logger(logger_)
        if logger_ is not None:
            try:
                loc = self._local
                q = loc.ExtractRotationQuat()
                self._entity = logger_.log_camera(
                    f"cameras/{self.name}",
                    width=self.width,
                    height=self.height,
                    focal_length_mm=self._focal_mm,
                    h_aperture_mm=self._h_aperture_mm,
                    v_aperture_mm=self._v_aperture_mm,
                    local_translation=list(loc.ExtractTranslation()),
                    local_quat_xyzw=[*q.GetImaginary(), q.GetReal()],
                    source=type(self).__name__,  # the Sensors instance tab carries the impl class
                )
            except Exception as exc:
                logger.warning(f"RtxCameraSensor {self.name}: pinhole logging unavailable: {exc!r}")

    def _grab_and_emit(self, t) -> None:
        prof = self._frame._prof
        ctx = prof.span(f"grab.{self.name}") if prof is not None else contextlib.nullcontext()
        with ctx:
            self._grab_and_emit_inner(t)

    def _grab_and_emit_inner(self, t) -> None:
        data = np.asarray(self._rgb.get_data())
        if data.ndim != 3 or data.size == 0:
            return  # cold pipeline: skip, keep the loop alive
        rgb = data[:, :, :3]
        if self._logger is not None:
            # the camera entity is a static child of vehicle/body; no per-frame transform needed
            self._logger.log_image(getattr(self, "_entity", None) or f"cameras/{self.name}", rgb, sim_time=t)
        if self._publisher is not None:
            self._publisher.push(rgb)

    def close(self) -> None:
        if self._publisher is not None:
            self._publisher.close()
