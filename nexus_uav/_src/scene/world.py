"""The world's values: where the scene sits on Earth and what follows from it, as attributes of Newton's model.

The geodetic origin, the magnetic field, the air pressure at sea level and the temperature are properties
of where the scene sits, so the model holds them, as it holds gravity: custom attributes at world frequency
in the ``nexus`` namespace, each read from the scene's ``PhysicsScene`` prim under its ``NexusSceneAPI``
attribute name. A sensor reads them from the model as it reads its site, and nothing samples them per tick.
"""

from __future__ import annotations

from pathlib import Path

import warp as wp
from newton import Model, ModelBuilder

from nexus_uav._src.core import geomag

GRAVITY = 9.81
"""Gravitational acceleration, m/s^2: the one value the physics applies and the IMU reports."""

NAMESPACE = "nexus"
# Each value's key on the builder and the model, ``model.nexus.<name>``.
ORIGIN = "nexus:geodetic_origin"
FIELD = "nexus:magnetic_field"
PRESSURE = "nexus:sea_level_pressure"
TEMPERATURE = "nexus:temperature"
SCENE_SCHEMA = "NexusSceneAPI"
ORIGIN_ATTR = "nexus:geodeticOrigin"


def _world(name: str, dtype, usd: str, default=None) -> ModelBuilder.CustomAttribute:
    return ModelBuilder.CustomAttribute(
        name=name,
        frequency=Model.AttributeFrequency.WORLD,
        assignment=Model.AttributeAssignment.MODEL,
        dtype=dtype,
        default=default,
        namespace=NAMESPACE,
        usd_attribute_name=usd,
    )


def attributes() -> list[ModelBuilder.CustomAttribute]:
    """The world's values as Newton custom attributes: the origin as latitude, longitude and altitude in
    degrees and metres, the field in the world's axes in gauss, the pressure at sea level in hectopascals
    and the temperature in degrees Celsius, each under its ``NexusSceneAPI`` attribute name.
    """
    return [
        _world("geodetic_origin", wp.vec3d, ORIGIN_ATTR),
        _world("magnetic_field", wp.vec3, "nexus:magneticField"),
        _world("sea_level_pressure", wp.float32, "nexus:seaLevelPressure", 1013.25),
        _world("temperature", wp.float32, "nexus:temperature", 25.0),
    ]


def declare(builder: ModelBuilder) -> None:
    """Register the world's values on `builder`, before any ``add_usd``, so the importer reads each one a
    scene's ``PhysicsScene`` authors.
    """
    for attribute in attributes():
        builder.add_custom_attribute(attribute)


def origin(builder: ModelBuilder) -> tuple[float, float, float] | None:
    """The origin a scene authored on `builder`, or ``None`` when no scene did."""
    value = builder.custom_attributes[ORIGIN].values.get(0)
    return None if value is None else tuple(float(x) for x in value)


def derive(builder: ModelBuilder) -> None:
    """Fill the field from the origin where no scene authored one, after the last ``add_usd``: the World
    Magnetic Model (WMM) table PX4 checks against, as a North East Down vector turned into the world's
    axes, ``(n, -e, -d)``. A builder with no origin keeps the field's default.
    """
    field = builder.custom_attributes[FIELD]
    where = origin(builder)
    if 0 in field.values or where is None:
        return
    n, e, d = geomag.ned_field(where[0], where[1])
    field.values[0] = wp.vec3(n, -e, -d)


def missing_origin(scene_usd: str | Path) -> str:
    """Why the scene at `scene_usd` gave the model no origin: it has no ``PhysicsScene`` prim, and the message
    names the file, or its ``PhysicsScene`` authors none, and the message names the prim.
    """
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(scene_usd), Usd.Stage.LoadAll)
    scenes = [str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    if not scenes:
        return f"{scene_usd}: the scene holds no PhysicsScene prim; a scene authors one, with {SCENE_SCHEMA} and its {ORIGIN_ATTR}"
    return (
        f"{scenes[0]}: the scene's PhysicsScene authors no {ORIGIN_ATTR}; apply {SCENE_SCHEMA} and author the "
        "latitude, longitude and altitude its world origin sits at"
    )


__all__ = [
    "FIELD",
    "GRAVITY",
    "ORIGIN",
    "PRESSURE",
    "TEMPERATURE",
    "attributes",
    "declare",
    "derive",
    "missing_origin",
    "origin",
]
