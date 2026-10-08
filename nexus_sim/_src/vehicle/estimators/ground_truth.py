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
    estimate: wp.array(dtype=PoseTwist),
):
    """Copy body ``i``'s pose and twist into the estimate."""
    tf = body_q[i]
    vd = body_qd[i]  # [linear, angular]
    e = PoseTwist()
    e.position = wp.transform_get_translation(tf)
    e.orientation = wp.transform_get_rotation(tf)
    e.linear_velocity = wp.spatial_top(vd)
    e.angular_velocity = wp.spatial_bottom(vd)
    estimate[0] = e


class GroundTruthEstimator:
    """The passthrough estimator: its stage writes the base body's true pose and twist, in world axes, to the
    signal ``estimate``, with no noise and no delay. So a guidance or a controller that reads the estimate
    reads the values of the physics state, and one written against the estimate runs behind any estimator.
    """

    def __init__(self):
        self.estimate = Signal("estimate", PoseTwist, shape=(1,))

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
