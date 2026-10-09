"""The RTX camera: the electro-optical (EO) feed over a ``Camera`` prim the vehicle USD authored.

The Kit peer renders it into one render product with an ``LdrColor`` annotator; here on the host the
sensor writes the frame to its signal ``camera`` and logs it as the First Person View image.
"""

from __future__ import annotations

from nexus_sim._src.core import logger
from nexus_sim._src.core.schema import Image
from nexus_sim._src.core.signals import Signal

from .rtx_sensor import RtxMountedSensor


class RtxCameraSensor(RtxMountedSensor):
    """RTX camera sensor: the peer's color output -> the signal ``camera``, an :class:`Image`, and
    ``Logger.log_image`` at the row named after the signal, ``sim/vehicle/sensors/<name>/camera``, under
    the frustum at the sensor's own entity.

    Args:
        run: The run's values: the ``Camera`` prim, the model body it rides and the render link.
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
        # authored intrinsics -> the Rerun Pinhole for the frustum / field-of-view visualization; logged once on set_logger
        self._focal_mm = float(prim.GetAttribute("focalLength").Get() or 12.0)
        self._h_aperture_mm = float(prim.GetAttribute("horizontalAperture").Get() or 36.0)
        self._v_aperture_mm = float(prim.GetAttribute("verticalAperture").Get() or 0.0) or None
        super().__init__(run, rate=rate, out=Signal("camera", Image))

    def set_logger(self, logger_) -> None:
        super().set_logger(logger_)
        if logger_ is not None:
            try:  # a frustum that fails to log costs the frustum alone: the frames still land at their row
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
        self.out.write(Image(t_shown, rgb))
        if self._logger is not None:
            # the camera rides the body's pose through its one static transform; no per-frame transform needed
            self._logger.log_image(self.out.name, rgb, sim_time=t_shown)
