"""The sensors a vehicle declares: one applied schema on each sensor's mount prim, built into its class.

The vehicle's Universal Scene Description (USD) file is the one authority for its sensors. A sensor's
prim sits under the rigid body it rides, and that parent body is the mount. The registry maps the
prim's schema to a class, the schema's attributes are the class's keyword arguments, and the class
takes the run's values, a :class:`~nexus._src.core.interfaces.SensorRun`, as its first argument.

A class states the peers it requires in a ``requires`` tuple, such as ``("kit",)`` on the RTX sensors.
The build starts each required peer once and hands the class its link. No vehicle names such a peer.
"""

from __future__ import annotations

from pathlib import Path

from nexus._src.core.components import ComponentSpec, resolve_components
from nexus._src.core.interfaces import SensorRun
from nexus._src.core.registry import ComponentRegistry


def _is_sensor(cls: type) -> bool:
    """Whether `cls` implements the sensor seam: it states stages, and it isn't a controller, which connects."""
    return callable(getattr(cls, "stages", None)) and not callable(getattr(cls, "connect", None))


def sensor_specs(usd_path: str | Path, registry: ComponentRegistry | None = None) -> list[ComponentSpec]:
    """One record per sensor the vehicle file at `usd_path` declares, in the order of its prims.

    With no `registry`, the default one resolves each schema.

    Raises:
        ValueError: A prim applies a schema no class claims, or a schema that doesn't apply to its
            type; the message names the prim.
    """
    return [spec for spec in resolve_components(usd_path, registry) if _is_sensor(spec.cls)]


def requires(spec: ComponentSpec, peer: str) -> bool:
    """Whether the class of `spec` requires the peer named `peer`."""
    return peer in getattr(spec.cls, "requires", ())


def build_sensors(specs: list[ComponentSpec], *, usd_path: str | Path, model, seedtree, dt: float, site, link=None):
    """Build each sensor of `specs` from its class, the run's values and its schema's keyword arguments.

    Args:
        specs: The sensors to build, from :func:`sensor_specs`.
        usd_path: The vehicle file that declares them.
        model: The physics model, whose body labels are the prim paths of the vehicle's rigid bodies.
        seedtree: The run's seed tree; each sensor's seed comes from it and the sensor's prim path.
        dt: The control tick, seconds.
        site: Where the run flies and its ambient values.
        link: The link a class that requires a peer takes; a class that requires none gets ``None``.

    Returns:
        The sensors, in the order of `specs`, each named after its prim, in lower case.

    Raises:
        ValueError: A sensor's prim doesn't sit under a rigid body of the model; the message names the prim.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    bodies = [str(label) for label in model.body_label]
    coms = model.body_com.numpy()
    sensors = []
    for spec in specs:
        prim = stage.GetPrimAtPath(spec.prim)
        parent = str(prim.GetParent().GetPath())
        if parent not in bodies:
            raise ValueError(
                f"{spec.prim}: {spec.schema} declares a sensor, and its parent {parent} isn't a rigid body "
                "of the vehicle; put the sensor's prim under the body it rides"
            )
        local = UsdGeom.Xformable(prim).GetLocalTransformation()
        t = local.ExtractTranslation()
        q = local.ExtractRotationQuat()  # a rigid mount's rotation; a class that needs one checks the prim
        body = bodies.index(parent)
        run = SensorRun(
            seed=seedtree.seed_for(spec.prim),
            dt=dt,
            site=site,
            body=body,
            mount=(float(t[0]), float(t[1]), float(t[2])),
            rotation=(*(float(x) for x in q.GetImaginary()), float(q.GetReal())),
            com=tuple(float(x) for x in coms[body]),
            path=spec.prim,
            prim=prim,
            link=link if getattr(spec.cls, "requires", ()) else None,
        )
        sensor = spec.cls(run, **spec.kwargs)
        # The instance is the prim that declares it: its name, in lower case, ends a recorded row's path
        # and keys the Recorder's channel, and its path is what a shared-name error names.
        sensor.name = spec.prim.rsplit("/", 1)[-1].lower()
        sensor.prim_path = spec.prim
        sensors.append(sensor)
    return sensors
