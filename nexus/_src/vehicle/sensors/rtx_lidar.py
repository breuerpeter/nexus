"""The RTX lidar over the authored ``OmniLidar`` prim.

A render product plus the ``IsaacExtractRTXSensorPointCloud`` annotator: Cartesian points and a
sensor-to-world transform, emitted as world-frame points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nexus._src.core import logger

from .rtx_sensor import RtxMountedSensor, _prim_sensor_attr

if TYPE_CHECKING:  # the frame is duck-typed at runtime; this is the annotation only
    from nexus._src.rendering.frame import RtxFrame


class RtxLidarSensor(RtxMountedSensor):
    """RTX lidar over the authored ``OmniLidar`` prim: a render product + the
    ``IsaacExtractRTXSensorPointCloud`` annotator, Cartesian points + a sensor-to-world transform,
    emitted as world-frame ``Points3D`` via ``Logger.log_points``, at ``lidar/<name>``.

    Full-scan accumulation is renderer-native: ``omni:sensor:Core:accumulateOutputs``, with
    ``tickRate == scanRateBaseHz`` baked at authoring; the scan advances on the
    ``/ExternalSimulationTime`` clock :class:`RtxFrame` drives from the lockstep sim time.
    """

    KIND = "lidar"

    def __init__(self, frame: RtxFrame, prim_path: str):
        import omni.usd

        self._annotator = None  # constructed in the pre-lockstep quiet window; see _ensure_sensor
        self._warned_no_data = False
        frame.quiet_window_hooks.append(self._ensure_sensor)
        # Sample rate from the AUTHORED prim, sensor:rate_hz, one sample per full scan; authoring
        # bakes the scan's own tickRate. 10 Hz fallback for legacy assets.
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
        super().__init__(frame, prim_path, rate_hz=float(_prim_sensor_attr(prim, "rate_hz", 10.0, "RtxLidarSensor")))

    def _ensure_sensor(self) -> None:
        """Attach the render product + annotator in the pre-lockstep quiet window;
        mid-flight construction races the captured graph's CUDA work, error 700.
        """
        if self._annotator is not None:
            return
        import omni.replicator.core as rep

        self._rp = rep.create.render_product(self.prim_path, (128, 128))
        self._annotator = rep.AnnotatorRegistry.get_annotator("IsaacExtractRTXSensorPointCloud")
        self._annotator.attach(self._rp)
        logger.info(f"RtxLidarSensor: {self.prim_path} pipeline attached")

    def _grab_and_emit(self, t) -> None:
        if self._annotator is None:
            return  # constructed in the pre-lockstep quiet window
        d = self._annotator.get_data()
        pts = d.get("data") if isinstance(d, dict) else d
        if pts is None:
            if isinstance(d, dict) and not self._warned_no_data:
                self._warned_no_data = True
                logger.warning(f"RtxLidarSensor {self.name}: no 'data' key in annotator output {list(d.keys())}")
            return
        pts = np.asarray(pts)
        if pts.size == 0 or pts.size % 3:
            return  # cold / between scans, or a malformed buffer; never die mid-flight
        pts = pts.reshape(-1, 3)
        # sensor frame -> world via the annotator's own transform. Raw reshape, NO .T: the Replicator
        # info transform is row-vector, translation in the last row, applied as pts @ M[:3,:3] + M[3,:3].
        info = d.get("info", {}) if isinstance(d, dict) else {}
        M = np.asarray(info["transform"], dtype=np.float64).reshape(4, 4) if "transform" in info else None
        if M is not None and abs(M[3, 3] - 1.0) <= 1e-3:
            pts = pts @ M[:3, :3] + M[3, :3]
        elif self._displayed_world is not None:  # fallback: this sensor's epoch mount matrix
            w = self._displayed_world
            r = np.asarray([[w[i][j] for j in range(3)] for i in range(3)])
            pts = pts @ r + np.asarray([w[3][0], w[3][1], w[3][2]])
        if len(pts) > 20000:  # subsample for the recording: a dense scan is ~270k pts @10 Hz -> GB rrds
            pts = pts[:: len(pts) // 20000 + 1]
        if self._logger is not None:
            self._logger.log_points(f"lidar/{self.name}", pts.astype(np.float32), radii=0.02)
