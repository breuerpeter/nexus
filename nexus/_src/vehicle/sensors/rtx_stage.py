"""The stage side of the RTX sensors: which prims a vehicle authored, and where they sit on the render stage.

Pure Universal Scene Description (USD) work on the host, no Kit: the vehicle USD is the single
authority on what renders, so the sensors, the renderer factory and the run's start decision all
read the vehicle through these functions rather than each walking it their own way.
"""

from __future__ import annotations

from nexus._src.core import logger

# The vehicle's root on the render stage: its own root prim, never under the scene's root, which
# carries the scene's -start translate; the vehicle composes at the world origin.
VEHICLE_ROOT = "/Vehicle"


def prim_modality(prim) -> str:
    """The camera prim's authored ``sensor:modality``, lowercased; none = ``"eo"``, so every
    vehicle authored before the attr existed keeps its exact behavior, with no fallback warning.
    """
    attr = prim.GetAttribute("sensor:modality")
    if not (attr and attr.HasAuthoredValue()):
        return "eo"
    value = str(attr.Get()).lower()
    if value not in ("eo", "ir"):
        logger.warning(f"{prim.GetPath()}: unknown sensor:modality {value!r}, rendering it EO")
        return "eo"
    return value


def discover_rtx_prims(stage, root: str) -> dict[str, list[str]]:
    """RTX-sensor prims authored under ``root``, the vehicle's root prim; the vehicle USD decides
    what renders, no flags: ``{"camera": [...], "lidar": [...]}``. Scoped to the vehicle on purpose:
    a camera the scene authors, a Cesium tile-selection camera or a scan's baked capture rig, isn't
    a vehicle sensor.
    """
    from pxr import Usd

    out: dict[str, list[str]] = {"camera": [], "lidar": []}
    prim = stage.GetPrimAtPath(root)
    if not prim:
        return out
    for p in Usd.PrimRange(prim):
        tn = p.GetTypeName()
        if tn == "Camera":
            out["camera"].append(str(p.GetPath()))
        elif tn == "OmniLidar":
            out["lidar"].append(str(p.GetPath()))
    return out


def render_path(path: str, root: str) -> str:
    """A vehicle prim's path on the render stage, where the vehicle's root prim sits at ``/Vehicle``."""
    return VEHICLE_ROOT + str(path)[len(str(root)) :]
