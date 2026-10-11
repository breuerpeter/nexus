"""The scene's geodetic origin: where its world origin sits on Earth, which its `PhysicsScene` prim authors.

A scene declares where it sits, as it declares its physics: `NexusSceneAPI` on its one `PhysicsScene` prim
authors `nexus:geodeticOrigin`. An override layer composed over the scene moves it. A scene that authors no
origin fails the build, so no run flies at an origin nobody chose.
"""

from __future__ import annotations

from pathlib import Path

SCENE_SCHEMA = "NexusSceneAPI"


def geodetic_origin(scene_usd: str | Path) -> tuple[float, float, float]:
    """The latitude and longitude in degrees and the altitude in metres that the scene at `scene_usd` authors.

    Raises:
        ValueError: The scene holds no `PhysicsScene` prim or more than one, and the message names the file;
            or its `PhysicsScene` authors no origin, and the message names the prim.
    """
    from pxr import Usd, UsdPhysics

    from nexus_sim._src.usd.reader import read_declarations

    stage = Usd.Stage.Open(str(scene_usd), Usd.Stage.LoadAll)
    scenes = [str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    if len(scenes) != 1:
        found = f"{len(scenes)} PhysicsScene prims, {', '.join(scenes)}" if scenes else "no PhysicsScene prim"
        raise ValueError(
            f"{scene_usd}: the scene holds {found}; a scene authors one, with {SCENE_SCHEMA} and its "
            "nexus:geodeticOrigin"
        )
    (path,) = scenes
    origins = [kwargs["geodetic_origin"] for prim, schema, kwargs in read_declarations(scene_usd) if (prim, schema) == (path, SCENE_SCHEMA)]
    origin = origins[0] if origins else None
    if origin is None:
        raise ValueError(
            f"{path}: the scene's PhysicsScene authors no nexus:geodeticOrigin; apply {SCENE_SCHEMA} and author "
            "the latitude, longitude and altitude its world origin sits at"
        )
    lat, lon, alt = origin
    return float(lat), float(lon), float(alt)


__all__ = ["SCENE_SCHEMA", "geodetic_origin"]
