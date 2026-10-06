"""The stage side of the RTX sensors: where a vehicle's prims sit on the render stage.

Pure Universal Scene Description (USD) work on the host, no Kit.
"""

from __future__ import annotations

# The vehicle's root on the render stage: its own root prim, never under the scene's root, which
# carries the scene's -start translate; the vehicle composes at the world origin.
VEHICLE_ROOT = "/Vehicle"


def render_path(path: str, root: str) -> str:
    """A vehicle prim's path on the render stage, where the vehicle's root prim sits at ``/Vehicle``."""
    return VEHICLE_ROOT + str(path)[len(str(root)) :]
