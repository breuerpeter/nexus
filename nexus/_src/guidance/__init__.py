"""Guidance: the outer loop of the control cascade, which turns a mission into the setpoint a
controller tracks.

A guidance is a component of the loop, for a controller that takes setpoints. It states a stage that
runs before the controller's, so the setpoint applies on the tick that computes it.
``MissionGuidance`` sequences position goals and advances on arrival. ``TrackingGuidance`` plans one
reference over the whole path and hands it to a tracking controller.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nexus._src.core.interfaces import Controller

__all__ = ["MissionGuidance", "TrackingGuidance"]


class MissionGuidance:
    """Sequence a mission of position goals for a controller that takes setpoints."""

    def __init__(
        self,
        controller: Controller,
        *,
        reached_m: float = 0.3,
        final_hold_s: float = 2.0,
        stop: Callable[[], None] | None = None,
        body_index: int = 0,
    ): ...


class TrackingGuidance:
    """Plan one reference over a mission's whole path and hand it to a tracking controller."""

    def __init__(
        self,
        controller: Controller,
        *,
        planner: Callable,
        final_hold_s: float = 2.0,
        stop: Callable[[], None] | None = None,
        body_index: int = 0,
    ): ...
