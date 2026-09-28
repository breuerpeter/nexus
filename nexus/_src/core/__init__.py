"""Newton framework core: schema, interfaces, capabilities, the eager
orchestrator, SeedTree, Scenario loader, plugin registry.

The transform between the world frame and the North East Down (NED) and Forward Right Down (FRD)
frames lives in ``nexus._src.transform``, not here: it's the one warp-dependent leaf, and
keeping it out of ``core`` is what lets ``core``, and so ``import nexus``, stay backend-agnostic
and importable without warp.
"""

from . import interfaces
from .clock import Clock
from .config import deep_merge
from .logging import logger
from .orchestrator import Orchestrator
from .registry import Registry
from .schema import (
    Controls,
    Measurement,
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
    "Measurement",
    "Orchestrator",
    "PositionGoal",
    "ReferenceTrajectory",
    "Registry",
    "SeedTree",
    "Setpoint",
    "SimTime",
    "Waypoints",
    "deep_merge",
    "interfaces",
    "logger",
]
