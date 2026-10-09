"""The RTX lidar over the authored ``OmniLidar`` prim.

The Kit peer renders it into a render product with the ``IsaacExtractRTXSensorPointCloud``
annotator and returns the scan's points in the world frame; here on the host the sensor writes them to
its signal ``lidar`` and logs them.
"""

from __future__ import annotations

import numpy as np

from nexus_sim._src.core.schema import PointCloud
from nexus_sim._src.core.signals import Signal

from .rtx_sensor import RtxMountedSensor


class RtxLidarSensor(RtxMountedSensor):
    """RTX lidar over the authored ``OmniLidar`` prim: the peer's world-frame points -> the signal
    ``lidar``, a :class:`PointCloud`, and ``Points3D`` via ``Logger.log_points``, at the row named after the
    signal, ``sim/vehicle/sensors/<name>/lidar``, under an entity that carries no transform, since the
    points are in world coordinates.

    Full-scan accumulation is renderer-native: ``omni:sensor:Core:accumulateOutputs``, with
    ``tickRate == scanRateBaseHz`` baked at authoring; the scan advances on the
    ``/ExternalSimulationTime`` clock the peer drives from the host's sim time.

    Args:
        run: The run's values: the ``OmniLidar`` prim, the model body it rides and the render link.
        rate: How often the lidar gives a full scan, hertz; authoring bakes the scan's own tick rate.
    """

    KIND = "lidar"
    output = "points"
    width = height = 128  # the render product the annotator reads; the scan pattern is the prim's own

    def __init__(self, run, rate: float = 10.0):
        super().__init__(run, rate=rate, out=Signal("lidar", PointCloud))

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
