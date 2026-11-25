"""Resolve the components a vehicle Universal Scene Description (USD) file declares through nexus schemas.

Each component's schema states its role, which places the component: a sensor under the body it rides, and
each controller, estimator and guidance on a `Scope` of its own whose parent is the vehicle's root prim. A
vehicle declares one or more of each, and at least one controller.
"""

from __future__ import annotations

import pathlib

from nexus_uav._src.core.components import ComponentSpec, resolve_components
from nexus_uav._src.core.registry import Registry

__all__ = ["SCOPE_ROLES", "ComponentSpec", "resolve_components", "scope_specs"]

# The roles whose components have no geometry: each sits on a `Scope` of its own under the vehicle's root prim.
SCOPE_ROLES = ("controller", "estimator", "guidance")


def scope_specs(usd_path: str | pathlib.Path, role: str, registry: Registry | None = None) -> list[ComponentSpec]:
    """Each component the vehicle file at `usd_path` declares in `role`, one of :data:`SCOPE_ROLES`, in the order
    of their prims, which is the order their stages run in.

    With no `registry`, the default one resolves each schema.

    Raises:
        ValueError: A component of `role` sits off a `Scope` whose parent is the root prim, the file's default
            prim, or two share one `Scope`, or the vehicle declares no controller; the message names the prim and
            its schemas and says where a component of `role` sits, or names the root prim.
    """
    from pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    default = stage.GetDefaultPrim().GetPath()  # empty with no default prim
    root = str(default) or str(usd_path)
    specs = [spec for spec in resolve_components(usd_path, registry) if spec.role == role]
    on: dict[str, str] = {}  # the schema of `role` each prim applies
    for spec in specs:
        path = Sdf.Path(spec.prim)
        on_root = default.isEmpty or path.GetParentPath() == default
        if not (on_root and stage.GetPrimAtPath(path).IsA(UsdGeom.Scope)):
            raise ValueError(
                f"{spec.prim}: {spec.schema} declares a {role}, and a {role} sits on a Scope of its own whose "
                f"parent is the vehicle's root prim {root}; put its Scope there"
            )
        if spec.prim in on:
            raise ValueError(
                f"{spec.prim}: {on[spec.prim]} and {spec.schema} both declare a {role}, and each {role} sits on "
                f"a Scope of its own under the vehicle's root prim {root}; give one a Scope of its own"
            )
        on[spec.prim] = spec.schema
    if role == "controller" and not specs:
        raise ValueError(
            f"{root}: the vehicle declares no controller; apply one, such as NexusPx4API_1, to a Scope under this prim"
        )
    return specs
