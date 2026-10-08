"""The stand-in project's estimator class, which `StandInEstimatorAPI` builds."""

import warp as wp

from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.core.schema import PoseTwist
from nexus_sim._src.core.signals import Signal


@wp.kernel
def _write_pose(position: wp.vec3, estimate: wp.array(dtype=PoseTwist)):
    """Write `position`, the identity orientation and zero twist into the estimate."""
    e = PoseTwist()
    e.position = position
    e.orientation = wp.quat_identity()
    estimate[0] = e


class StandInEstimator:
    """An estimator whose device stage writes a fixed pose to the estimate: its position, the identity
    orientation and zero twist.

    Args:
        position: The schema's `nexus:position`, in world axes.
    """

    def __init__(self, position=(0.0, 0.0, 0.0)):
        self.position = wp.vec3(*(float(x) for x in position))
        self.estimate = Signal("estimate", PoseTwist, shape=(1,))

    def stages(self) -> list[Stage]:
        """One device stage, which writes the fixed pose to the estimate."""
        return [Stage("estimate", "device", self._run, writes=(self.estimate,))]

    def _run(self, tick) -> None:
        wp.launch(_write_pose, dim=1, inputs=(self.position,), outputs=(self.estimate.buffer,))
