"""The RTX camera: the electro-optical (EO) feed over a ``Camera`` prim the vehicle USD authored.

The Kit peer renders it into one render product with an ``LdrColor`` annotator; here on the host the
sensor logs the frame as the First Person View image and, when the launch asks for a stream, pushes
it to the Real Time Streaming Protocol (RTSP) publisher.
"""

from __future__ import annotations

from nexus._src.core import logger

from .rtsp import RtspPublisher
from .rtx_sensor import RtxMountedSensor, _prim_sensor_attr


class RtxCameraSensor(RtxMountedSensor):
    """RTX camera sensor: the peer's color output -> ``Logger.log_image`` at ``cameras/<name>`` and,
    when streaming, the :class:`RtspPublisher`.

    Args:
        link: The render link.
        prim: The ``Camera`` prim on the vehicle's own stage.
        path: The prim's path on the render stage.
        body: The model body index the camera rides.
        cfg: The RTX settings: resolution and rate fallbacks for a prim that authors none, and the
            stream bitrate.
        stream_url: Where to publish the feed, or ``None`` for the recording only.
    """

    KIND = "cameras"
    output = "color"

    def __init__(self, link, prim, *, path: str, body: int, cfg, stream_url: str | None = None):
        # Render params from the AUTHORED prim, sensor:width/height/rate_hz; the vehicle USD is
        # the single authority, and the RTX settings only cover legacy assets / explicit rtx: overrides.
        self.width = int(_prim_sensor_attr(prim, "width", cfg.width, "RtxCameraSensor"))
        self.height = int(_prim_sensor_attr(prim, "height", cfg.height, "RtxCameraSensor"))
        rate_hz = float(_prim_sensor_attr(prim, "rate_hz", cfg.render_hz, "RtxCameraSensor"))
        # The --stream consumer is per camera at the CAMERA's resolution/rate.
        self._publisher = (
            RtspPublisher(
                width=self.width,
                height=self.height,
                fps=max(1, round(rate_hz)),
                bitrate=cfg.bitrate,
                rtsp_url=stream_url,
            )
            if stream_url
            else None
        )
        # authored intrinsics -> the Rerun Pinhole for the frustum / field-of-view visualization; logged once on set_logger
        self._focal_mm = float(prim.GetAttribute("focalLength").Get() or 12.0)
        self._h_aperture_mm = float(prim.GetAttribute("horizontalAperture").Get() or 36.0)
        self._v_aperture_mm = float(prim.GetAttribute("verticalAperture").Get() or 0.0) or None
        self._entity = None
        super().__init__(link, prim, path=path, body=body, rate_hz=rate_hz)

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

    def emit(self, arrays: dict, t_shown: float) -> None:
        rgb = arrays.get("color")
        if rgb is None:
            return
        if self._logger is not None:
            # the camera entity is a static child of vehicle/body; no per-frame transform needed
            self._logger.log_image(self._entity or f"cameras/{self.name}", rgb, sim_time=t_shown)
        if self._publisher is not None:
            self._publisher.push(rgb)

    def close(self) -> None:
        if self._publisher is not None:
            self._publisher.close()
