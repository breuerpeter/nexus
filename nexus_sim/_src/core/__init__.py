"""The nexus core: schema, interfaces, capabilities, the eager
orchestrator, SeedTree, Scenario loader, component registry.

Core runs on Warp: its schema declares the device signals' types as Warp structs. It imports no physics
backend, so ``newton`` stays out of it. The transform between the world frame and the
North East Down (NED) and Forward Right Down (FRD) frames lives in ``nexus_sim._src.transform``.
"""

from . import interfaces
from .clock import Clock
from .config import deep_merge
from .logging import logger
from .orchestrator import Orchestrator
from .schema import (
    Controls,
    PositionGoal,
    ReferenceTrajectory,
    Setpoint,
    SimTime,
    Waypoints,
)
from .seedtree import SeedTree

__all__ = [
    "Clock",
    "Controls",
    "Orchestrator",
    "PositionGoal",
    "ReferenceTrajectory",
    "SeedTree",
    "Setpoint",
    "SimTime",
    "Waypoints",
    "deep_merge",
    "interfaces",
    "logger",
]
