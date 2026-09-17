"""The Long Wave Infrared (LWIR) band law and the Wiris display chain: pure numpy, so no Kit and no GPU.

The law is the contract #60's scene builder authors emission against and the thermal sensor
inverts, so the test runs it as a round trip. The display chain test covers its two visible
promises: fixed radiometric stops, where the same temperature always draws the same pixel, and a hard
color step at the fire threshold.
"""

import numpy as np
import pytest

from nexus._src.vehicle.sensors.lwir import (
    intensity_from_temperature,
    lwir_display,
    temperature_from_intensity,
)

# The shipped stops, mirrored from RtxThermalSensor in nexus/_src/vehicle/sensors/rtx_thermal.py.
DISPLAY = {
    "t_lo": 300.0,
    "t_hi": 340.0,
    "ramp_t_lo": 400.0,
    "ramp_t_hi": 1500.0,
    "ramp_stops": ((0.00, (255, 0, 200)), (0.33, (255, 110, 0)), (0.66, (255, 215, 0)), (1.00, (255, 255, 225))),
    "ambient_gray": 0.10,
    "ambient_sigma": 0.02,
    "noise_sigma": 0.008,
}


def _display(temps, sky=None):
    """Render a 1-D row of kelvin, where 0 = not emissive, through the shipped stops, noise-free."""
    T = np.asarray(temps, dtype=np.float32).reshape(1, -1)
    sky = np.zeros(T.shape, dtype=bool) if sky is None else np.asarray(sky, dtype=bool).reshape(1, -1)
    return lwir_display(T, T > 0.0, sky, rng=None, **DISPLAY)[0]


def test_law_round_trips_within_1_kelvin():
    T = np.linspace(250.0, 1800.0, 200)
    assert np.max(np.abs(temperature_from_intensity(intensity_from_temperature(T)) - T)) < 1.0


def test_law_anchors_match_the_scene_calibration():
    # The anchors #60 authors against, and the sensor's SAT_INTENSITY = 12000 white clamp.
    assert intensity_from_temperature(1500.0) == pytest.approx(11595.0, rel=0.01)
    assert intensity_from_temperature(300.0) == pytest.approx(165.0, rel=0.02)


def test_warm_temperatures_are_distinct_gray_and_hot_ones_are_colour():
    px = _display([305.0, 330.0, 900.0, 1500.0])
    warm, hot = px[:2], px[2:]
    assert warm[0][0] != warm[1][0], "305 K and 330 K must not display the same level"
    assert all(len(set(p.tolist())) == 1 for p in warm), "below the ramp the triplet is gray"
    assert all(len(set(p.tolist())) > 1 for p in hot), "900 K and 1500 K must carry colour"


def test_the_ramp_step_is_hard():
    below, above = _display([DISPLAY["ramp_t_lo"] - 1.0, DISPLAY["ramp_t_lo"] + 1.0])
    assert len(set(below.tolist())) == 1, "just below the threshold is still a gray triplet"
    assert len(set(above.tolist())) > 1, "just above it the colour ramp takes over"


def test_ambient_floor_lifts_unheated_ground_but_never_dims_a_hot_pixel():
    floor = int(DISPLAY["ambient_gray"] * 255.0)  # the uint8 cast truncates
    assert _display([0.0])[0].tolist() == [floor] * 3
    assert _display([330.0])[0][0] > floor


def test_sky_is_exactly_black():
    px = _display([0.0, 330.0, 1500.0], sky=[True, True, True])
    assert np.count_nonzero(px) == 0


def test_noise_is_opt_in_and_deterministic_without_a_generator():
    T = np.full((8, 8), 320.0, dtype=np.float32)
    flags = np.ones(T.shape, dtype=bool)
    clean = lwir_display(T, flags, ~flags, rng=None, **DISPLAY)
    assert np.all(clean == clean[0, 0])
    grainy = lwir_display(T, flags, ~flags, rng=np.random.default_rng(0), **DISPLAY)
    assert not np.all(grainy == grainy[0, 0])
