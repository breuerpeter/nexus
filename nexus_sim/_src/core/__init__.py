"""The nexus core: the component contract, the loop, the signal types, the clock and the registry.

Core runs on Warp: its schema declares the device signals' types as Warp structs. It imports no physics
backend, so ``newton`` stays out of it. The transform between the world frame and the
North East Down (NED) and Forward Right Down (FRD) frames lives in ``nexus_sim._src.transform``.
"""

from . import interfaces
from .clock import Clock
from .config import deep_merge
from .interfaces import Component
from .logging import logger
from .loop import Loop
from .registry import Registry, register
from .schema import (
    PositionGoal,
    ReferenceTrajectory,
    Setpoint,
    SimTime,
    Waypoints,
)
from .seedtree import SeedTree

__all__ = [
    "Clock",
    "Component",
    "Loop",
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
    "register",
]
