"""The scene temperature law and the Long Wave Infrared (LWIR) display chain, per #54 and #60.

One documented, invertible law shared by the scene author and the thermal sensor:

    emissive_intensity = K_INTENSITY_PER_RADIANCE * L_band(T)

where ``L_band(T)`` is the blackbody radiance integrated over the LWIR window, 8-14 um,
in W m^-2 sr^-1; Planck, emissivity 1. Band integration, not raw Stefan-Boltzmann sigma*T^4,
because an LWIR core only responds to this window: the 1500 K / 316 K contrast is 56x in-band compared
to 286x total, which keeps the whole scene inside the one OmniPBR intensity range the renderer-transfer
measurement covered. The FaaS paper models radiative heat via Stefan-Boltzmann energy
particles and has no camera-side formula; the band law here is the camera-side adaptation.

Anchors with K = 3.0, chosen so the hottest authored state hits the sensor's measured
saturation constant:  1500 K -> 11595, close to SAT_INTENSITY 12000; 316 K sun-warmed rock -> 208;
300 K ambient -> 165.

:func:`lwir_display` is the other half: kelvin -> the 8-bit image for the operator, with the
Wiris-style ramp over a fixed threshold. It lives here, beside the law, because it's pure
numpy and so testable without Kit; :class:`~nexus._src.vehicle.sensors.rtx_thermal.
RtxThermalSensor` is the thin binding that inverts the renderer transfer and calls it.

Pure numpy on purpose: imported by the sensor post in the container and the scene builder on the host.
"""

from __future__ import annotations

import numpy as np

_H, _C, _KB = 6.62607015e-34, 2.99792458e8, 1.380649e-23
BAND_M = (8.0e-6, 14.0e-6)  # the LWIR window
K_INTENSITY_PER_RADIANCE = 3.0  # authored emissive_intensity units per W m^-2 sr^-1


def band_radiance(T, lo: float = BAND_M[0], hi: float = BAND_M[1], n: int = 512):
    """Blackbody radiance integrated over [lo, hi] wavelengths: W m^-2 sr^-1. Vectorized in T."""
    T = np.asarray(T, dtype=np.float64)
    lam = np.linspace(lo, hi, n)
    B = 2.0 * _H * _C**2 / lam[None, :] ** 5 / np.expm1(_H * _C / (lam[None, :] * _KB * T.reshape(-1, 1)))
    out = np.trapezoid(B, lam, axis=1)
    return out.reshape(np.shape(T)) if np.shape(T) else float(out[0])


def intensity_from_temperature(T):
    """The scene law: authored ``emissive_intensity`` for a surface at T kelvin."""
    return K_INTENSITY_PER_RADIANCE * band_radiance(T)


# Inversion Lookup Table (LUT), monotonic: 200..2000 K covers frost to flame core.
_LUT_T = np.linspace(200.0, 2000.0, 1801)
_LUT_I = intensity_from_temperature(_LUT_T)


def temperature_from_intensity(intensity):
    """Invert the law, numpy-vectorized. Zero/negative intensity -> 200 K floor."""
    return np.interp(np.asarray(intensity, dtype=np.float64), _LUT_I, _LUT_T)


def lwir_display(
    T,
    emissive,
    sky,
    *,
    t_lo: float,
    t_hi: float,
    ramp_t_lo: float,
    ramp_t_hi: float,
    ramp_stops,
    ambient_gray: float,
    ambient_sigma: float,
    noise_sigma: float,
    rng=None,
) -> np.ndarray:
    """Kelvin -> the 8-bit LWIR image: white-hot gray, with a color ramp over a hard threshold.

    Radiometric on purpose, the stops are constant temperatures, not a scene-adaptive Automatic Gain
    Control (AGC), so two runs of the same flight are directly comparable and the scene's authored
    temperatures land where the author put them.

    Args:
        T: ``(H, W)`` float kelvin; the value is only read where *emissive* is True.
        emissive: ``(H, W)`` bool; pixels the scene authored emission on. Everything else has
            no radiometric temperature at all, only the synthesized ambient floor.
        sky: ``(H, W)`` bool; dome pixels, forced to black, since a clear LWIR sky is the cold end.
        t_lo: Grayscale floor in kelvin; displays black.
        t_hi: Grayscale ceiling in kelvin; displays white.
        ramp_t_lo: Color onset in kelvin. The step here is hard: no blend band.
        ramp_t_hi: Kelvin at the top color stop.
        ramp_stops: ``((u, (r, g, b)), ...)``, u in [0, 1] ascending, channels in [0, 255].
        ambient_gray: The floor every non-sky pixel gets, for surfaces the scene authors no
            emission on. Applied as a lower bound, so a hot pixel is never dimmed by it.
        ambient_sigma: Gaussian spread on that floor; needs *rng*.
        noise_sigma: NETD-style grain added to the whole gray image last; needs *rng*.
        rng: ``np.random.Generator``; None means no noise at all: deterministic, for tests.

    Returns:
        ``(H, W, 3)`` uint8.
    """
    T = np.asarray(T, dtype=np.float32)
    emissive = np.asarray(emissive, dtype=bool)
    sky = np.asarray(sky, dtype=bool)
    gray = np.zeros(T.shape, dtype=np.float32)
    gray[emissive] = np.clip((T[emissive] - t_lo) / max(t_hi - t_lo, 1e-6), 0.0, 1.0)
    # Ambient floor first, sky after it: the dome must stay black even where the floor would lift it.
    amb = np.full(T.shape, float(ambient_gray), dtype=np.float32)
    if rng is not None and ambient_sigma:
        amb = amb + float(ambient_sigma) * rng.standard_normal(T.shape).astype(np.float32)
    gray = np.maximum(gray, np.where(sky, 0.0, amb))
    gray[sky] = 0.0
    if rng is not None and noise_sigma:
        gray = gray + float(noise_sigma) * rng.standard_normal(T.shape).astype(np.float32)
    u8 = (np.clip(gray, 0.0, 1.0) * 255.0).astype(np.uint8)
    img = np.repeat(u8[..., None], 3, axis=2)
    # The Wiris ramp REPLACES the gray triplet over the threshold: a hard step, so the operator
    # reads "this is fire" at a glance rather than judging a shade of white.
    hot = emissive & ~sky & (T >= ramp_t_lo)
    if hot.any():
        u = np.clip((T[hot] - ramp_t_lo) / max(ramp_t_hi - ramp_t_lo, 1e-6), 0.0, 1.0)
        xp = np.asarray([s[0] for s in ramp_stops], dtype=np.float32)
        for ch in range(3):
            fp = np.asarray([s[1][ch] for s in ramp_stops], dtype=np.float32)
            img[hot, ch] = np.clip(np.interp(u, xp, fp), 0.0, 255.0).astype(np.uint8)
    return img
