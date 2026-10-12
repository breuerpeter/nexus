"""The sensors a vehicle declares: one applied schema on each sensor's mount prim, built into its class.

The vehicle's Universal Scene Description (USD) file is the one authority for its sensors. A sensor's
schema states the sensor role. Its prim sits under the rigid body it rides, and that parent body is the
mount. The registry maps the prim's schema to a class, and the class takes the model and the prim's path,
the label of the site Newton loaded the prim as, then the sensor's noise seed and the schema's attributes
as keyword arguments. The class reads its body and its mount from the model, so the builder places nothing.

A class states the peers it requires in a ``requires`` tuple, such as ``("kit",)`` on the RTX sensors.
The build starts each required peer once, and the loop hands the class the peer's link at ``connect``.
No vehicle names such a peer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nexus_sim._src.core.components import ComponentSpec, resolve_components
from nexus_sim._src.core.registry import Registry


def sensor_specs(usd_path: str | Path, registry: Registry | None = None) -> list[ComponentSpec]:
    """One record per sensor the vehicle file at `usd_path` declares, in the order of its prims.

    With no `registry`, the default one resolves each schema.

    Raises:
        ValueError: A prim applies a schema no class claims, a schema that doesn't apply to its type, or
            a schema that states no role or more than one; the message names the prim.
    """
    return [spec for spec in resolve_components(usd_path, registry) if spec.role == "sensor"]


def requires(spec: ComponentSpec, peer: str) -> bool:
    """Whether the class of `spec` requires the peer named `peer`."""
    return peer in spec.cls.requires


def _rigid_mount(prim, schema: str) -> None:
    """Fail a sensor whose prim's transform scales, shears or mirrors it: a mount only moves and turns.

    Raises:
        ValueError: The prim's transform isn't rigid; the message names the prim.
    """
    from pxr import UsdGeom

    m = np.array(UsdGeom.Xformable(prim).GetLocalTransformation())[:3, :3]
    if not np.allclose(m @ m.T, np.eye(3), atol=1e-6) or np.linalg.det(m) < 0.0:
        raise ValueError(
            f"{prim.GetPath()}: {schema} declares a sensor, whose mount only moves and turns, and this prim's "
            "transform scales, shears or mirrors it; author a translation and a rotation alone"
        )


def build_sensors(specs: list[ComponentSpec], *, usd_path: str | Path, model, seedtree):
    """Build each sensor of `specs` from its class: the model and its prim's path, then its seed and its
    schema's keyword arguments.

    Args:
        specs: The sensors to build, from :func:`sensor_specs`.
        usd_path: The vehicle file that declares them, for the sensor role's place checks.
        model: The physics model, whose body labels are the prim paths of the vehicle's rigid bodies and
            whose site labels are the paths of the sensors' prims.
        seedtree: The run's seed tree; each sensor's seed comes from it and the sensor's prim path.

    Returns:
        The sensors, in the order of `specs`, each named after its prim, in lower case.

    Raises:
        ValueError: A sensor's prim doesn't sit under a rigid body of the model, its transform scales, shears
            or mirrors the mount, or two sensors' prims share a name; the message names the prims.
    """
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    bodies = [str(label) for label in model.body_label]
    named: dict[str, str] = {}
    for spec in specs:
        name = spec.prim.rsplit("/", 1)[-1].lower()
        if name in named:
            raise ValueError(
                f"sensors {named[name]} and {spec.prim} share the name {name!r}: a recorded row's path ends in "
                "its sensor's name, so rename one prim"
            )
        named[name] = spec.prim
    sensors = []
    for spec in specs:
        prim = stage.GetPrimAtPath(spec.prim)
        parent = str(prim.GetParent().GetPath())
        if parent not in bodies:
            raise ValueError(
                f"{spec.prim}: {spec.schema} declares a sensor, and its parent {parent} isn't a rigid body "
                "of the vehicle; put the sensor's prim under the body it rides"
            )
        _rigid_mount(prim, spec.schema)
        sensor = spec.cls(model, spec.prim, seed=seedtree.seed_for(spec.prim), **spec.kwargs)
        # The instance is the prim that declares it: its name, in lower case, ends a recorded row's path
        # and keys the Recorder's history.
        sensor.name = spec.prim.rsplit("/", 1)[-1].lower()
        sensors.append(sensor)
    return sensors
