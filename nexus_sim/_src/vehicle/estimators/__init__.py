"""Estimators: the role whose stage runs after the sensors' and before the guidance's, and writes the
vehicle's estimate, which the guidance and the controllers read.
"""

from .ground_truth import GroundTruthEstimator

__all__ = ["GroundTruthEstimator"]
