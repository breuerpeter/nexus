"""The RTX camera: the electro-optical (EO) feed over a ``Camera`` prim the vehicle USD authored.

The render stage renders it into one render product with an ``LdrColor`` render var; here on the host the
sensor writes the frame to its signal ``camera`` and logs it as the First Person View image.
"""

from __future__ import annotations

from nexus_uav._src.core import logger
from nexus_uav._src.core.schema import Image
from nexus_uav._src.core.signals import Signal

from .sensor import RtxMountedSensor


class RtxCameraSensor(RtxMountedSensor):
    """RTX camera sensor: the render stage's color output -> the signal ``camera``, an :class:`Image`, and
    ``Logger.log_image`` at the row named after the signal, ``sim/vehicle/sensors/<name>/camera``, under
    the frustum at the sensor's own entity.

    Args:
        model: Newton's model, whose body the ``Camera`` prim sits under the sensor rides.
        path: The path of the ``Camera`` prim on the vehicle's stage.
        seed: The sensor's noise seed, which a camera doesn't draw on.
        width: Width of the image, pixels.
        height: Height of the image, pixels.
        rate: How often the camera gives a frame, hertz.
    """

    KIND = "cameras"
    output = "color"

    def __init__(self, model, path: str, *, seed: int = 0, width: int = 1280, height: int = 720, rate: float = 24.0):
        self.width = int(width)
        self.height = int(height)
        self._focal_mm = 12.0
        self._h_aperture_mm = 36.0
        self._v_aperture_mm: float | None = None
        super().__init__(model, path, seed=seed, rate=rate, out=Signal("camera", Image))

    def _read_prim(self, prim) -> None:
        """The authored intrinsics -> the Rerun Pinhole for the frustum / field-of-view visualization, logged
        once, here at connect, with the mount read beside it.
        """
        self._focal_mm = float(prim.GetAttribute("focalLength").Get() or 12.0)
        self._h_aperture_mm = float(prim.GetAttribute("horizontalAperture").Get() or 36.0)
        self._v_aperture_mm = float(prim.GetAttribute("verticalAperture").Get() or 0.0) or None
        if self._logger is not None:
            try:  # a frustum that fails to log costs the frustum alone: the frames still land at their row
                loc = self._local
                q = loc.ExtractRotationQuat()
                self._logger.log_camera(
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
