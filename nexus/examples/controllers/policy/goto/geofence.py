"""GeofenceGuidance: the worked example of a guidance specialisation, for the goto policy flight.

A mission guidance with a box around the flight. When the vehicle leaves the box, the guidance keeps
where and when it happened, freezes the mission, logs a red marker at the breach, and ends the run. The flight
reports the breach in its stats, so a policy that strays fails for that reason, not only for the
waypoints it then misses.
"""

from __future__ import annotations

import numpy as np

from nexus._src.guidance import MissionGuidance

__all__ = ["GeofenceGuidance"]

_FENCE = (255, 140, 0)  # the box's rings
_KILL = (255, 0, 0)  # the marker at the breach
_KILL_RADIUS = 0.3  # [m]


class GeofenceGuidance(MissionGuidance):
    """A mission guidance that ends the run when the vehicle leaves its box.

    The stage reads the vehicle's position once per tick, as the mission guidance does, and checks it
    against the box first. On a breach it sets ``breached_at`` and ``breach_pos``, freezes the mission
    through ``_done``, logs the breach marker and calls ``stop``. A new mission clears the breach, so the
    fence fires once per mission rather than once per instance.

    Args:
        controller: The controller that takes setpoints.
        bounds: The box as ``((x_min, y_min, z_min), (x_max, y_max, z_max))`` in world axes [m].
        **mission: :class:`MissionGuidance`'s own arguments: ``reached_m``, ``final_hold_s``, ``stop``
            and ``body_index``.
    """

    def __init__(self, controller, *, bounds, **mission):
        super().__init__(controller, **mission)
        lo, hi = bounds
        self._lo = np.asarray(lo, dtype=float)
        self._hi = np.asarray(hi, dtype=float)
        self._fenced = False  # the first tick logs the rings, once
        self.breached_at: float | None = None
        """The sim time of the breach, or ``None`` while the vehicle has stayed inside the box."""
        self.breach_pos: tuple[float, float, float] | None = None
        """The vehicle's world position at the breach, or ``None``."""

    def set_mission(self, setpoints) -> None:
        """Set the mission and clear an earlier breach, so the fence fires once per mission."""
        self.breached_at = None
        self.breach_pos = None
        super().set_mission(setpoints)

    def _tick(self, pos: np.ndarray, ts: float) -> None:
        if not self._fenced:
            self._fenced = True
            self._log_fence()
        if self.breached_at is not None:
            return
        if np.any(pos < self._lo) or np.any(pos > self._hi):
            self.breached_at = ts
            self.breach_pos = (float(pos[0]), float(pos[1]), float(pos[2]))
            self._done = True
            if self._logger is not None:
                self._logger.log_points("guidance/breach", [list(self.breach_pos)], colors=_KILL, radii=_KILL_RADIUS)
            self._end()
            return
        super()._tick(pos, ts)

    def _log_fence(self) -> None:
        """Log the box as two rings, its floor and its ceiling, under ``guidance/fence``."""
        if self._logger is None:
            return
        (x0, y0, z0), (x1, y1, z1) = self._lo, self._hi
        for name, z in (("floor", z0), ("ceiling", z1)):
            ring = [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z], [x0, y0, z]]
            self._logger.log_strip(f"guidance/fence/{name}", ring, color=_FENCE, radius=0.02)
