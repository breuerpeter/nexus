"""The RTX thermal camera: the Long Wave Infrared (LWIR) feed over a camera prim marked ``ir``.

The Kit peer renders the same kind of render product as the electro-optical camera and returns its
radiance and depth; here on the host the radiance turns into kelvin, then into the operator's 8-bit
image, through the one invertible band law in :mod:`nexus._src.vehicle.sensors.lwir`.
"""

from __future__ import annotations

import numpy as np

from nexus._src.core import logger

from .rtsp import RtspPublisher
from .rtx_sensor import RtxMountedSensor, _prim_sensor_attr


class RtxThermalSensor(RtxMountedSensor):
    """RTX thermal sensor, Long Wave Infrared (LWIR): the peer's radiance, the ``PtSelfIllumination``
    Arbitrary Output Variable (AOV) on the authored camera prim, -> an 8-bit white-hot image via
    ``Logger.log_image``, at ``cameras/<name>``, and, when streaming, the :class:`RtspPublisher`.

    The scene encodes temperature as OmniPBR emission, the band law in
    :mod:`~nexus._src.vehicle.sensors.lwir`, with ``emissive_color.r = 1.0`` so red
    radiance = 0.318 x ``emissive_intensity``, measured; this post inverts that transfer, masks
    sky by depth, since a High Dynamic Range Image (HDRI) textured dome floods the AOV, inverts the
    law to kelvin and displays it against fixed radiometric stops, so the same authored temperature
    always draws the same pixel and two runs of one flight compare directly.

    Selected by ``sensor:modality = "ir"`` on the camera prim; with none authored, EO -> RtxCameraSensor.
    Two constraints the authoring and the construction order have to respect:

    * the Pt AOV returns at the DLSS INTERNAL resolution, so author ``sensor:width/height`` at
      twice the wanted IR core: 1280x1024 -> 640x512 under the pinned Performance mode, measured;
    * the render product must exist before the first rendered frame, created with the EO cameras;
      a Pt-annotated product created after frames have rendered stays permanently empty, measured.
    """

    KIND = "cameras"
    output = "radiance_depth"
    RADIANCE_PER_INTENSITY = 0.318  # measured OmniPBR transfer, ~1/pi, stable across 6.0.0/6.0.1
    SAT_INTENSITY = 12000.0  # white clamp ~= 1520 K under the scene law, thermal.py, K=3.0
    SKY_DEPTH_M = 20000.0  # beyond the terrain ring -> the dome
    DISPLAY_T_LO = 300.0  # grayscale floor, kelvin
    DISPLAY_T_HI = 340.0  # grayscale ceiling, kelvin
    RAMP_T_LO = 400.0  # color onset, kelvin
    RAMP_T_HI = 1500.0  # top color, kelvin
    RAMP_STOPS = ((0.00, (255, 0, 200)), (0.33, (255, 110, 0)),
                  (0.66, (255, 215, 0)), (1.00, (255, 255, 225)))  # fmt: skip
    AMBIENT_GRAY = 0.10  # synthesized floor for surfaces the scene authors no emission on
    AMBIENT_SIGMA = 0.02
    NOISE_SIGMA = 0.008  # NETD-style grain

    def __init__(self, link, prim, *, path: str, body: int, cfg, stream_url: str | None = None):
        # Render params from the AUTHORED prim, exactly as RtxCameraSensor does; the vehicle USD is
        # the single authority. What comes back is smaller, the DLSS-internal grid, so nothing
        # downstream of the render can assume these numbers; see _emit_size.
        self.width = int(_prim_sensor_attr(prim, "width", 1280, "RtxThermalSensor"))
        self.height = int(_prim_sensor_attr(prim, "height", 1024, "RtxThermalSensor"))
        rate_hz = float(_prim_sensor_attr(prim, "rate_hz", cfg.render_hz, "RtxThermalSensor"))
        # authored intrinsics -> the Rerun Pinhole; logged on the first frame, at the EMITTED size
        self._focal_mm = float(prim.GetAttribute("focalLength").Get() or 12.0)
        self._h_aperture_mm = float(prim.GetAttribute("horizontalAperture").Get() or 36.0)
        self._v_aperture_mm = float(prim.GetAttribute("verticalAperture").Get() or 0.0) or None
        self._bitrate = cfg.bitrate
        self._stream_url = stream_url
        self._publisher = None  # also first-frame: the encoder needs the emitted size, not the authored one
        self._entity = None
        self._rng = np.random.default_rng(0)
        super().__init__(link, prim, path=path, body=body, rate_hz=rate_hz)

    def set_logger(self, logger_) -> None:
        """Take the Logger, but defer the pinhole: it needs the emitted image size, and no frame
        has rendered yet when the orchestrator hands the Logger over.
        """
        super().set_logger(logger_)
        self._entity = None

    def _emit_size(self, shape) -> None:
        """First-frame binding of everything that needs the EMITTED size, the AOV's own grid, not
        the authored render-product size: the Rerun pinhole and the RTSP encoder. Idempotent.
        """
        h, w = int(shape[0]), int(shape[1])
        if self._entity is None and self._logger is not None:
            try:
                loc = self._local
                q = loc.ExtractRotationQuat()
                self._entity = self._logger.log_camera(
                    f"cameras/{self.name}",
                    width=w,
                    height=h,
                    focal_length_mm=self._focal_mm,
                    h_aperture_mm=self._h_aperture_mm,
                    v_aperture_mm=self._v_aperture_mm,
                    local_translation=list(loc.ExtractTranslation()),
                    local_quat_xyzw=[*q.GetImaginary(), q.GetReal()],
                    source=type(self).__name__,
                )
            except Exception as exc:
                self._entity = f"cameras/{self.name}"  # log images anyway; only the frustum goes missing
                logger.warning(f"RtxThermalSensor {self.name}: pinhole logging unavailable: {exc!r}")
        if self._publisher is None and self._stream_url:
            self._publisher = RtspPublisher(
                width=w,
                height=h,
                fps=max(1, round(self.rate_hz)),
                bitrate=self._bitrate,
                rtsp_url=self._stream_url,
            )

    def emit(self, arrays: dict, t_shown: float) -> None:
        rad = arrays.get("radiance")
        if rad is None:
            return
        self._emit_size(rad.shape)
        img = self._post(rad, self._sky_mask(rad.shape, arrays.get("depth")))
        if self._logger is not None:
            self._logger.log_image(self._entity or f"cameras/{self.name}", img, sim_time=t_shown)
        if self._publisher is not None:
            self._publisher.push(img)

    def _sky_mask(self, shape, depth) -> np.ndarray:
        """Dome pixels on the AOV grid, True = sky, from the full-res depth buffer, strided down."""
        if depth is None or depth.size == 0:
            return np.zeros(shape, dtype=bool)
        d = depth.reshape(depth.shape[0], depth.shape[1]).astype(np.float32)
        if d.shape != tuple(shape):
            sy, sx = max(d.shape[0] // shape[0], 1), max(d.shape[1] // shape[1], 1)
            d = d[::sy, ::sx][: shape[0], : shape[1]]
            if d.shape != tuple(shape):
                return np.zeros(shape, dtype=bool)
        return ~np.isfinite(d) | (d > self.SKY_DEPTH_M)

    def _post(self, rad: np.ndarray, sky: np.ndarray) -> np.ndarray:
        """Emission radiance -> the (H, W, 3) uint8 LWIR image. The thin binding: undo the renderer
        transfer to authored-intensity units, undo the scene law to kelvin, then hand the field to
        :func:`~nexus._src.vehicle.sensors.lwir.lwir_display`.

        The AOV is a hot-sources-over-ambient channel, not full radiometry; it's exactly zero
        wherever the scene authors no emission, so the display synthesizes the ambient floor
        rather than measuring it. Per-material ambient IR is #67.
        """
        from .lwir import lwir_display, temperature_from_intensity

        intensity = rad * (1.0 / self.RADIANCE_PER_INTENSITY)  # authored emissive_intensity units
        intensity[sky] = 0.0
        np.clip(intensity, 0.0, self.SAT_INTENSITY, out=intensity)
        emissive = intensity > 0.0
        T = np.zeros_like(intensity)
        if emissive.any():
            T[emissive] = temperature_from_intensity(intensity[emissive]).astype(np.float32)
        return lwir_display(
            T,
            emissive,
            sky,
            t_lo=self.DISPLAY_T_LO,
            t_hi=self.DISPLAY_T_HI,
            ramp_t_lo=self.RAMP_T_LO,
            ramp_t_hi=self.RAMP_T_HI,
            ramp_stops=self.RAMP_STOPS,
            ambient_gray=self.AMBIENT_GRAY,
            ambient_sigma=self.AMBIENT_SIGMA,
            noise_sigma=self.NOISE_SIGMA,
            rng=self._rng,
        )

    def close(self) -> None:
        if self._publisher is not None:
            self._publisher.close()
