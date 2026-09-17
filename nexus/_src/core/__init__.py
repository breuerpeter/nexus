"""Newton framework core: schema, interfaces, capabilities, the eager
orchestrator, SeedTree, ConstantEnvironment, Scenario loader, plugin registry.

The transform between the world frame and the North East Down (NED) and Forward Right Down (FRD)
frames lives in ``nexus._src.transform``, not here: it's the one warp-dependent leaf, and
keeping it out of ``core`` is what lets ``core``, and so ``import nexus``, stay backend-agnostic
and importable without warp. The Isaac Sim runtime needs that, since Kit only puts warp on the path
after SimulationApp boots.
"""

from . import interfaces
from .clock import Clock
from .config import deep_merge
from .environment import ConstantEnvironment
from .logging import logger
from .orchestrator import Orchestrator
from .registry import Registry
from .schema import (
    Controls,
    EnvSample,
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
    "ConstantEnvironment",
    "Controls",
    "EnvSample",
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
