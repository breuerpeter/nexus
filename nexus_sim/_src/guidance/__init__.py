"""Guidance: the outer loop of the control cascade, which turns a mission into the setpoint a
controller tracks.

A guidance is a component of the loop, for a controller that takes setpoints. It states a stage that
runs before the controller's, so the setpoint applies on the tick that computes it.
``MissionGuidance`` sequences position goals and advances on arrival. ``TrackingGuidance`` plans one
reference over the whole path for a tracking controller. Each writes its setpoint to a signal the
controller reads. A flight constructs its
guidance and hands it to ``Sim.from_orchestrator``; PX4 flies its own missions, so a PX4 run has none.
"""

from .base import Guidance
from .mission import MissionGuidance
from .tracking import TrackingGuidance

__all__ = ["Guidance", "MissionGuidance", "TrackingGuidance"]
