"""The RTX camera: the electro-optical (EO) feed over a ``Camera`` prim the vehicle USD authored.

The Kit peer renders it into one render product with an ``LdrColor`` annotator; here on the host the
sensor logs the frame as the First Person View image and, when the launch asks for a stream, pushes
it to the Real Time Streaming Protocol (RTSP) publisher.
"""

from __future__ import annotations

from nexus._src.core import logger

from .rtsp import RtspPublisher
from .rtx_sensor import RtxMountedSensor


class RtxCameraSensor(RtxMountedSensor):
    """RTX camera sensor: the peer's color output -> ``Logger.log_image`` at the sensor's own entity,
    ``sim/vehicle/sensors/<name>``, and, when streaming, the :class:`RtspPublisher`.

    Args:
        run: The run's values: the ``Camera`` prim, the model body it rides and the render link, which
            says where to publish the feed when the run streams.
        width: Width of the image, pixels.
        height: Height of the image, pixels.
        rate: How often the camera gives a frame, hertz.
    """

    KIND = "cameras"
    output = "color"

    def __init__(self, run, width: int = 1280, height: int = 720, rate: float = 24.0):
        prim = run.prim
        self.width = int(width)
        self.height = int(height)
        # The --stream consumer is per camera at the CAMERA's resolution/rate.
        stream_url = run.link.streams.url("cam")
        self._publisher = (
            RtspPublisher(
                width=self.width,
                height=self.height,
                fps=max(1, round(rate)),
                bitrate=run.link.streams.bitrate,
                rtsp_url=stream_url,
            )
            if stream_url
            else None
        )
        # authored intrinsics -> the Rerun Pinhole for the frustum / field-of-view visualization; logged once on set_logger
        self._focal_mm = float(prim.GetAttribute("focalLength").Get() or 12.0)
        self._h_aperture_mm = float(prim.GetAttribute("horizontalAperture").Get() or 36.0)
        self._v_aperture_mm = float(prim.GetAttribute("verticalAperture").Get() or 0.0) or None
        super().__init__(run, rate=rate)

    def set_logger(self, logger_) -> None:
        super().set_logger(logger_)
        if logger_ is not None:
            try:  # a frustum that fails to log costs the frustum alone: the frames still land at this entity
                loc = self._local
                q = loc.ExtractRotationQuat()
                logger_.log_camera(
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
            # the camera rides the body's pose through its one static transform; no per-frame transform needed
            self._logger.log_image("", rgb, sim_time=t_shown)
        if self._publisher is not None:
            self._publisher.push(rgb)

    def close(self) -> None:
        if self._publisher is not None:
            self._publisher.close()
