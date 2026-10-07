"""GroundTruthEstimator: the passthrough estimator, which hands on the base body's true pose and twist."""

from __future__ import annotations

from nexus_sim._src.core.interfaces import Stage


class GroundTruthEstimator:
    """The passthrough estimator: its stage writes the base body's true pose and twist, in world axes, to the
    signal ``estimate``, with no noise and no delay.
    """

    def stages(self) -> list[Stage]:
        """One warm device stage, which copies the base body's row of the physics state into the estimate."""
        ...


__all__ = ["GroundTruthEstimator"]
