"""The RTX lidar over the authored ``OmniLidar`` prim.

The Kit peer renders it into a render product with the ``IsaacExtractRTXSensorPointCloud``
annotator and returns the scan's points in the world frame; here on the host the sensor logs them.
"""

from __future__ import annotations

import numpy as np

from .rtx_sensor import RtxMountedSensor, _prim_sensor_attr


class RtxLidarSensor(RtxMountedSensor):
    """RTX lidar over the authored ``OmniLidar`` prim: the peer's world-frame points -> ``Points3D``
    via ``Logger.log_points``, at ``lidar/<name>``.

    Full-scan accumulation is renderer-native: ``omni:sensor:Core:accumulateOutputs``, with
    ``tickRate == scanRateBaseHz`` baked at authoring; the scan advances on the
    ``/ExternalSimulationTime`` clock the peer drives from the host's sim time.

    Args:
        link: The render link.
        prim: The ``OmniLidar`` prim on the vehicle's own stage.
        path: The prim's path on the render stage.
        body: The model body index the lidar rides.
    """

    KIND = "lidar"
    output = "points"
    width = height = 128  # the render product the annotator reads; the scan pattern is the prim's own

    def __init__(self, link, prim, *, path: str, body: int):
        # Sample rate from the AUTHORED prim, sensor:rate_hz, one sample per full scan; authoring
        # bakes the scan's own tickRate. 10 Hz fallback for legacy assets.
        rate_hz = float(_prim_sensor_attr(prim, "rate_hz", 10.0, "RtxLidarSensor"))
        super().__init__(link, prim, path=path, body=body, rate_hz=rate_hz)

    def emit(self, arrays: dict, t_shown: float) -> None:
        pts = arrays.get("points")
        if pts is None or self._logger is None:
            return
        if len(pts) > 20000:  # subsample for the recording: a dense scan is ~270k pts @10 Hz -> GB rrds
            pts = pts[:: len(pts) // 20000 + 1]
        # Stamped at the time the scan shows, a frame or two before this tick, then the tick's own
        # time back for the rest of its logging.
        self._logger.log_points(f"lidar/{self.name}", np.asarray(pts, dtype=np.float32), radii=0.02, sim_time=t_shown)
        self._logger.set_time(self._now)
