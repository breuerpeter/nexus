"""ConstantEnvironment: authoritative ambient fields, as architecture.md §11 describes.

Greenfield in the framework: the bridge had no Environment, only a scalar
ref_alt plus a magnetic field hardcoded inside the messaging layer. v1 is a
constant provider supplying gravity plus the World Magnetic Model (WMM) magnetic field that the
magnetometer, and hence PX4's heading estimate, depends on.

The field *must* match the HIL_GPS origin, or PX4's strict mag arming check, which compares against PX4's
own internal WMM table, fails "Strong magnetic interference" and refuses to arm. Use :meth:`from_gps`: it
computes declination/inclination/strength from the Global Positioning System (GPS) origin via the **same**
coarse WMM table PX4 uses, :mod:`nexus._src.core.geomag`, ported from PX4 ``geo_mag_declination.cpp``,
so the simulated field matches PX4's expectation exactly. The bare ``__init__`` keeps explicit field params,
default the bridge's Zurich field, for the non-PX4 paths, policy, Proportional Integral Derivative (PID) and
Model Predictive Control (MPC), where nothing reads the magnetometer.
"""

from __future__ import annotations

import math

from . import geomag
from .schema import EnvSample


def _wmm_ned(declination_deg: float, inclination_deg: float, strength_gauss: float) -> tuple[float, float, float]:
    """The geomagnetic field as a North East Down (NED) vector from declination D, inclination Inc, strength F.

    Horizontal intensity H = F·cos(Inc); the NED components are then N = H·cos(D), E = H·sin(D),
    down = F·sin(Inc). The earlier form dropped the cos(Inc) factor from E and carried a spurious
    cos(D) on the down component: harmless at small declination, but at larger |D| it skewed both
    the inclination and the declination of the resulting field, which PX4's strict WMM arming check
    rejects as "magnetic interference"; the total strength stayed the same either way.
    """
    decl = math.radians(declination_deg)
    incl = math.radians(inclination_deg)
    horizontal = strength_gauss * math.cos(incl)
    mag_n = horizontal * math.cos(decl)
    mag_e = horizontal * math.sin(decl)
    mag_d = strength_gauss * math.sin(incl)
    return (mag_n, mag_e, mag_d)


class ConstantEnvironment:
    def __init__(
        self,
        gravity_world: tuple[float, float, float] = (0.0, 0.0, -9.81),
        declination_deg: float = 3.0,  # bridge: Zurich
        inclination_deg: float = 64.0,
        strength_gauss: float = 0.48,
    ):
        self._sample = EnvSample(
            gravity_world=gravity_world,
            mag_ned=_wmm_ned(declination_deg, inclination_deg, strength_gauss),
        )

    @classmethod
    def from_gps(
        cls,
        lat_deg: float,
        lon_deg: float,
        *,
        gravity_world: tuple[float, float, float] = (0.0, 0.0, -9.81),
    ) -> ConstantEnvironment:
        """Build the environment with the WMM field at the GPS origin, computed from PX4's own coarse
        WMM table in :mod:`geomag` so PX4's strict mag arming check passes exactly. The PX4 assembly
        uses this with ``cfg.sensors.gps.init`` so the field tracks the origin.
        """
        decl, incl, strength = geomag.field_at(lat_deg, lon_deg)
        return cls(
            gravity_world=gravity_world,
            declination_deg=decl,
            inclination_deg=incl,
            strength_gauss=strength,
        )

    def sample(self, pos, t) -> EnvSample:
        return self._sample
