"""GroundTruthEstimator: the passthrough estimator, which hands on the base body's true pose and twist."""

from __future__ import annotations

import warp as wp

from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.core.schema import PoseTwist
from nexus_sim._src.core.signals import Signal


@wp.kernel
def copy_pose_twist(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    i: int,
    estimate: wp.array2d(dtype=float),
):
    """Copy body ``i``'s pose and twist into the estimate's one row, in the order of a body's row in the Recorder."""
    tf = body_q[i]
    p = wp.transform_get_translation(tf)
    q = wp.transform_get_rotation(tf)  # xyzw
    vd = body_qd[i]  # [lin(0:3), ang(3:6)]
    estimate[0, 0] = p[0]
    estimate[0, 1] = p[1]
    estimate[0, 2] = p[2]
    estimate[0, 3] = q[0]
    estimate[0, 4] = q[1]
    estimate[0, 5] = q[2]
    estimate[0, 6] = q[3]
    estimate[0, 7] = vd[0]
    estimate[0, 8] = vd[1]
    estimate[0, 9] = vd[2]
    estimate[0, 10] = vd[3]
    estimate[0, 11] = vd[4]
    estimate[0, 12] = vd[5]


class GroundTruthEstimator:
    """The passthrough estimator: its stage writes the base body's true pose and twist, in world axes, to the
    signal ``estimate``, with no noise and no delay. So a guidance or a controller that reads the estimate
    reads the values of the physics state, and one written against the estimate runs behind any estimator.
    """

    def __init__(self):
        self.estimate = Signal("estimate", PoseTwist, shape=(1, 13))

    def stages(self) -> list[Stage]:
        """One warm device stage, ``estimate``, which copies the base body's row of the physics state into the
        estimate. The loop runs it after the sensors' stages and before the guidance's, and once over the
        settled state before the first tick, so the guidance's first stage reads an estimate.
        """
        return [Stage("estimate", "device", self._run, writes=(self.estimate,))]

    def _run(self, tick) -> None:
        wp.launch(
            copy_pose_twist,
            dim=1,
            inputs=(tick.state.body_q, tick.state.body_qd, tick.base),
            outputs=(self.estimate.buffer,),
        )


__all__ = ["GroundTruthEstimator"]
