"""GroundTruthEstimator: the passthrough estimator, which hands on a body's true pose and twist."""

from __future__ import annotations

import warp as wp

from nexus_uav._src.core.interfaces import Component, Stage
from nexus_uav._src.core.schema import PoseTwist
from nexus_uav._src.core.signals import Signal


class GroundTruthEstimator(Component):
    """The passthrough estimator: its stage copies the pose and twist its ``pose`` input holds, the plant's
    output for the base body unless a connection on its prim names another body, to the signal ``estimate``,
    in world axes, with no noise and no delay. So a guidance or a controller that reads the estimate reads
    the values of the physics state, and one written against the estimate runs behind any estimator.
    """

    def __init__(self):
        self.pose = Signal("pose", PoseTwist, shape=(1,))
        self.estimate = Signal("estimate", PoseTwist, shape=(1,))

    def stages(self) -> list[Stage]:
        """One warm device stage, ``estimate``, which copies the pose into the estimate. The loop runs it after
        the sensors' stages and before the guidance's, and once over the settled state before the first tick,
        so the guidance's first stage reads an estimate.
        """
        return [Stage("estimate", "device", self._run, reads=(self.pose,), writes=(self.estimate,))]

    def _run(self, tick) -> None:
        wp.copy(self.estimate.buffer, self.pose.buffer)


__all__ = ["GroundTruthEstimator"]
