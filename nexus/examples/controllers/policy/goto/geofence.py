"""GeofenceGuidance: the worked example of a guidance specialisation, for the goto policy flight.

A mission guidance with a box around the flight. When the vehicle leaves the box, the guidance ends
the run and keeps where and when it happened.
"""

from __future__ import annotations

from nexus._src.guidance import MissionGuidance

__all__ = ["GeofenceGuidance"]


class GeofenceGuidance(MissionGuidance):
    """A mission guidance that ends the run when the vehicle leaves its box."""

    def __init__(self, controller, *, bounds, **mission): ...
