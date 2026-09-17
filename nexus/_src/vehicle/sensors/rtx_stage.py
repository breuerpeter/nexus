"""The stage side of the RTX sensors: which prims a vehicle authored, and the pose each write uses.

Pure Universal Scene Description (USD) work, no Kit: the vehicle USD is the single authority on
what renders, so the sensors, the render frame, the Isaac Sim launch and the cesium globe all read
the stage through these functions rather than each walking it their own way.
"""

from __future__ import annotations

from nexus._src.core import logger

# The vehicle's stage root: its own root prim, session-authored, never under the scene's root,
# which carries the runtime -start translate; the vehicle composes at the world origin.
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


def discover_rtx_prims(stage, root: str = VEHICLE_ROOT) -> dict[str, list[str]]:
    """RTX-sensor prims authored under the VEHICLE's stage subtree; the vehicle USD decides what
    renders, no flags: ``{"camera": [...], "lidar": [...]}``. Scoped to ``root`` deliberately:
    the scene owns the rest of the stage, and a scene-authored Camera, a cesium tile-selection
    camera or a scan's baked capture rig, isn't a vehicle sensor.
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


def pose_matrix(Gf, bq):
    """A body pose row of ``body_q`` (px py pz, qx qy qz qw) → a row-vector ``Gf.Matrix4d`` world
    matrix: the one conversion every Fabric pose write, vehicle body prims and mounted sensors, uses.
    """
    m = Gf.Matrix4d().SetRotate(Gf.Quatd(float(bq[6]), float(bq[3]), float(bq[4]), float(bq[5])))
    m.SetTranslateOnly(Gf.Vec3d(float(bq[0]), float(bq[1]), float(bq[2])))
    return m
