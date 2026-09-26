"""USD-authored analytic sensors: the vehicle Universal Scene Description (USD) is the single authority
for the sensor suite.

The analytic PX4 sensors ride in the vehicle USD the same way as the RTX ones, the ``Camera`` /
``OmniLidar`` prims, and the actuator params, the ``motor:*`` / ``propeller:*`` joint attrs: each sensor
is an ``Xform`` child of the base body carrying a ``sensor:type`` custom token, ``imu`` / ``mag`` /
``baro`` / ``gps``, plus its parameters as ``sensor:*`` custom attributes, named exactly after the
sensor constructors' keyword arguments, ``sensor:acc_noise`` → ``ImuSensor(acc_noise=...)``,
so a typo in the USD fails the build loudly. The prim's local translation is the body-frame mount offset,
which the Inertial Measurement Unit (IMU) consumes as its lever arm; the other kinds model no mount, so
author them at the body origin. World properties stay out of the vehicle USD: the geodetic origin of the
Global Positioning System (GPS) comes from the scene/scenario config. The vehicle declares that it has a
GPS, not where that GPS is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SENSOR_TYPE_ATTR = "sensor:type"
_PREFIX = "sensor:"


def vehicle_rtx_sensor_prims(usd_path: str | Path) -> list[str]:
    """Paths of the RTX-sensor prims under the vehicle's root prim, from a host-side ``usd-core`` scan.

    RTX sensors render in the Kit peer, so their presence is what starts it; a vehicle with none
    starts no container.
    """
    from pxr import Usd

    from .rtx_stage import discover_rtx_prims

    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    root = stage.GetDefaultPrim()
    if not root:
        return []
    prims = discover_rtx_prims(stage, str(root.GetPath()))
    return prims["camera"] + prims["lidar"]


@dataclass(frozen=True)
class SensorSpec:
    """One authored sensor: its kind, body-frame mount translation, and constructor kwargs."""

    kind: str  # imu | mag | baro | gps
    mount: tuple[float, float, float] = (0.0, 0.0, 0.0)
    params: dict = field(default_factory=dict)


def _plain(value):
    """A pxr attribute value as a plain Python value: Gf vectors → float tuples."""
    if hasattr(value, "__len__") and not isinstance(value, str):
        return tuple(float(x) for x in value)
    return value


def parse_sensor_prims(usd_path: str | Path) -> list[SensorSpec]:
    """The analytic sensors authored in the vehicle USD, from a host-side ``usd-core`` scan.

    Any prim with an authored ``sensor:type`` is a sensor; its other ``sensor:*`` attributes become
    the constructor kwargs and its local translation the mount offset. Returns specs in stage
    traversal order, which is authoring order; ``[]`` when the USD authors none.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    specs = []
    # Default predicate, which skips deactivated prims, the standard USD way to remove a sensor in an
    # override layer, + instance proxies, meaning sensors inside an instanceable vehicle reference.
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        type_attr = prim.GetAttribute(SENSOR_TYPE_ATTR)
        if not type_attr or not type_attr.HasAuthoredValue():
            continue
        params = {
            attr.GetName()[len(_PREFIX) :]: _plain(attr.Get())
            for attr in prim.GetAttributes()
            if attr.GetName().startswith(_PREFIX) and attr.GetName() != SENSOR_TYPE_ATTR and attr.HasAuthoredValue()
        }
        t = UsdGeom.Xformable(prim).GetLocalTransformation().ExtractTranslation()
        specs.append(SensorSpec(str(type_attr.Get()), (float(t[0]), float(t[1]), float(t[2])), params))
    return specs


# Not authorable as sensor:* attrs: the mount is the prim's translation; dt and the geodetic
# origin are properties of the run and the world, supplied by the assembly.
_RESERVED_PARAMS = ("mount_offset", "dt", "ref_lat", "ref_lon", "ref_alt")


def _sensor_registry() -> dict:
    """The ``sensor:type`` token → constructor registry, the selection mechanism, mirroring newton's
    actuator schema→class registry: the USD's token picks the implementation, its ``sensor:*``
    attrs are the constructor kwargs verbatim. A new implementation, for example a second IMU model, is
    one entry with a new token; the parsing/kwarg plumbing is already generic.
    """
    from .sensors import BaroSensor, GpsSensor, ImuSensor, MagSensor

    return {
        "imu": lambda spec, seedtree, dt, gps_init, ref_alt: ImuSensor(
            seedtree, dt, mount_offset=spec.mount, **spec.params
        ),
        "mag": lambda spec, seedtree, dt, gps_init, ref_alt: MagSensor(seedtree, **spec.params),
        "baro": lambda spec, seedtree, dt, gps_init, ref_alt: BaroSensor(seedtree, **spec.params),
        "gps": lambda spec, seedtree, dt, gps_init, ref_alt: GpsSensor(
            gps_init["lat"], gps_init["lon"], ref_alt, **spec.params
        ),
    }


def build_sensors(specs: list[SensorSpec], *, seedtree, dt: float, gps_init: dict, ref_alt: float) -> list:
    """Instantiate the sensor suite from its specs; ``gps_init``/``ref_alt`` = the scene's geodetic
    origin. Controller-agnostic: any assembly builds its suite this way. Raises on an
    unknown/duplicate kind, an unknown/reserved parameter, or a mount on a sensor that can't model
    one; a config bug fails the build, not the flight.
    """
    registry = _sensor_registry()
    sensors, seen = [], set()
    for spec in specs:
        if spec.kind in seen:  # duplicates would share the per-kind noise seed + Measurement fields
            raise ValueError(f"duplicate sensor:type {spec.kind!r}: author one prim per kind")
        seen.add(spec.kind)
        if reserved := [k for k in spec.params if k in _RESERVED_PARAMS]:
            raise ValueError(f"reserved sensor:* attrs {reserved} on {spec.kind!r} (see _RESERVED_PARAMS)")
        if spec.kind != "imu" and spec.mount != (0.0, 0.0, 0.0):  # only the IMU models a lever arm
            raise ValueError(f"sensor:type {spec.kind!r} cannot model a mount: author the prim at the body origin")
        make = registry.get(spec.kind)
        if make is None:
            raise ValueError(f"unknown sensor:type {spec.kind!r} (expected one of {sorted(registry)})")
        sensors.append(make(spec, seedtree, dt, gps_init, ref_alt))
    return sensors
