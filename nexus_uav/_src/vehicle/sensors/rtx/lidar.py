"""The RTX lidar over the authored ``OmniLidar`` prim.

The render stage renders it into a render product with a ``PointCloud`` render var and hands back the scan's
points in the world frame; here on the host the sensor writes them to its signal ``lidar`` and logs them.
"""

from __future__ import annotations

import numpy as np

from nexus_uav._src.core.schema import PointCloud
from nexus_uav._src.core.signals import Signal

from .sensor import RtxMountedSensor


class RtxLidarSensor(RtxMountedSensor):
    """RTX lidar over the authored ``OmniLidar`` prim: the render stage's world-frame points -> the signal
    ``lidar``, a :class:`PointCloud`, and ``Points3D`` via ``Logger.log_points``, at the row named after the
    signal, ``sim/vehicle/sensors/<name>/lidar``, under an entity that carries no transform, since the
    points are in world coordinates.

    The renderer completes a scan when its clock, the elapsed sim time each render step hands it, passes a period
    of ``omni:sensor:frameRate``, which the render stage authors from ``rate``, and hands it over a period later: a
    10 Hz scan that completes at 104 ms reaches the signal at the due tick at 204 ms, stamped with that tick's time.

    Args:
        model: Newton's model, whose body the ``OmniLidar`` prim sits under the sensor rides.
        path: The path of the ``OmniLidar`` prim on the vehicle's stage.
        seed: The sensor's noise seed, which a lidar doesn't draw on.
        rate: How often the lidar gives a full scan, hertz; authoring bakes the scan's own tick rate.
    """

    KIND = "lidar"
    output = "points"
    width = height = 128  # the render product's size; the scan pattern is the prim's own

    def __init__(self, model, path: str, *, seed: int = 0, rate: float = 10.0):
        super().__init__(model, path, seed=seed, rate=rate, out=Signal("lidar", PointCloud))

    def emit(self, arrays: dict, t_shown: float) -> None:
        pts = arrays.get("points")
        if pts is None:
            return
        self.out.write(PointCloud(t_shown, np.asarray(pts, dtype=np.float32)))
        if self._logger is None:
            return
        if len(pts) > 20000:  # subsample for the recording: a dense scan is ~270k pts @10 Hz -> GB rrds
            pts = pts[:: len(pts) // 20000 + 1]
        # Stamped at the time the scan shows, a frame or two before this tick, then the tick's own
        # time back for the rest of its logging.
        self._logger.log_points(self.out.name, np.asarray(pts, dtype=np.float32), radii=0.02, sim_time=t_shown)
        self._logger.set_time(self._now)
