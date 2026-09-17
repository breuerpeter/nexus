"""PX4-matching World Magnetic Model (WMM) table, core/geomag.py: the simulated field at the
Global Positioning System (GPS) origin must match PX4's own coarse WMM table, so PX4's strict mag
arming check passes with no "Strong magnetic interference." These pin the Seattle GPS origin, the
default scenario, and the from_gps wiring.
"""

import math

from nexus._src.core import ConstantEnvironment, geomag

# Seattle GPS origin from the default scenario, sensors.gps.init.
SEATTLE = (47.747944, -122.163917)


def test_field_at_seattle_matches_px4_table():
    decl, incl, strength = geomag.field_at(*SEATTLE)
    # PX4's coarse-table interpolation at the Seattle origin: the values PX4's check expects.
    assert 15.3 < decl < 15.8  # ~15.5 deg E declination
    assert 69.0 < incl < 69.8  # ~69.4 deg inclination
    assert 0.535 < strength < 0.543  # ~0.539 gauss, compared to the Zurich default's 0.48, which PX4 rejected


def test_from_gps_total_field_equals_table_strength():
    """from_gps's North East Down (NED) field norm equals the table strength, the quantity PX4's strength gate checks."""
    _, _, strength = geomag.field_at(*SEATTLE)
    n, e, d = ConstantEnvironment.from_gps(*SEATTLE).sample(None, None).mag_ned
    assert math.isclose(math.sqrt(n * n + e * e + d * d), strength, abs_tol=1e-4)


def test_lookup_in_range_and_out_of_range():
    assert geomag._lookup(0.0, 0.0, geomag._STRENGTH_TABLE) > 0  # equator: finite field
    assert geomag._lookup(200.0, 0.0, geomag._STRENGTH_TABLE) == 0.0  # out of range -> 0
