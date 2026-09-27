"""The site: the ambient values a run reads, resolved once at build from the scene's geodetic origin.

The magnetic field, the air pressure and temperature, and gravity are properties of where the scene
sits, so the build resolves them once and hands them to the sensors that read them as constructor
arguments. Nothing samples them per tick, so the captured graph sees no host value.
"""

from __future__ import annotations

from dataclasses import dataclass

from nexus._src.core import geomag

GRAVITY = 9.81
"""Gravitational acceleration, m/s^2: the one value the physics applies and the IMU reports."""


@dataclass(frozen=True, slots=True)
class Site:
    """Where a run flies, and the ambient values that follow from it.

    The world frame is Newton Forward Left Up (FLU), Z-up; ``mag_ned`` is in North East Down (NED).
    """

    lat: float
    """WGS84 latitude of the local origin, degrees."""
    lon: float
    """WGS84 longitude of the local origin, degrees."""
    alt: float
    """Altitude of the local origin above mean sea level, metres."""
    mag_ned: tuple[float, float, float]
    """Earth magnetic field at the origin as a NED vector, gauss."""
    pressure_msl: float = 1013.25
    """Air pressure at mean sea level, hPa."""
    temperature: float = 25.0
    """Ambient air temperature, degrees Celsius."""
    gravity: float = GRAVITY
    """Gravitational acceleration, m/s^2, along world -Z."""

    @classmethod
    def at(cls, lat: float, lon: float, alt: float) -> Site:
        """The site at a geodetic origin: its field from the World Magnetic Model (WMM) table PX4 checks
        against, so PX4's strict mag arming check passes exactly; the rest the standard atmosphere.
        """
        return cls(lat=float(lat), lon=float(lon), alt=float(alt), mag_ned=geomag.ned_field(lat, lon))


__all__ = ["GRAVITY", "Site"]
