"""Resolve the components a vehicle Universal Scene Description (USD) file declares through nexus schemas.

Each component's schema states its role, which places the component: a sensor under the body it rides, and
the controller and the estimator each on a `Scope` of its own whose parent is the vehicle's root prim.
"""

from __future__ import annotations

import pathlib

from nexus_sim._src.core.components import ComponentSpec, resolve_components
from nexus_sim._src.core.registry import ComponentRegistry

__all__ = ["ComponentSpec", "declared_controller", "declared_estimator", "prims_applying", "resolve_components"]


def _on_scopes(
    usd_path: str | pathlib.Path, role: str, registry: ComponentRegistry | None
) -> tuple[str, list[ComponentSpec]]:
    """The vehicle's root prim, or its file with no default prim, and each component it declares in `role`.

    Raises:
        ValueError: Two or more components of `role`, or one whose prim isn't a `Scope` whose parent is
            the root prim; the message names the root prim and each schema and prim, or the one prim
            and its schema and says where a component of `role` sits.
    """
    from pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    default = stage.GetDefaultPrim().GetPath()  # empty with no default prim
    root = str(default) or str(usd_path)
    specs = [spec for spec in resolve_components(usd_path, registry) if spec.role == role]
    if len(specs) > 1:
        placed = ", ".join(f"{spec.schema} on {spec.prim}" for spec in specs)
        raise ValueError(f"{root}: the vehicle declares {len(specs)} {role}s, {placed}; a vehicle declares one {role}")
    for spec in specs:
        path = Sdf.Path(spec.prim)
        on_root = default.isEmpty or path.GetParentPath() == default
        if not (on_root and stage.GetPrimAtPath(path).IsA(UsdGeom.Scope)):
            raise ValueError(
                f"{spec.prim}: {spec.schema} declares the {role}, and a {role} sits on a Scope of its own whose "
                f"parent is the vehicle's root prim {root}; put its Scope there"
            )
    return root, specs


def declared_controller(usd_path: str | pathlib.Path, registry: ComponentRegistry | None = None) -> ComponentSpec:
    """The one controller the vehicle file at `usd_path` declares, on the `Scope` that applies its schema.

    With no `registry`, the default one resolves each schema.

    Raises:
        ValueError: The vehicle declares no controller, more than one, or one off a `Scope` whose
            parent is its root prim, its default prim; the message names the root prim, and for two or
            more, their schemas and prims, or the prim off its place.
    """
    root, controllers = _on_scopes(usd_path, "controller", registry)
    if not controllers:
        raise ValueError(
            f"{root}: the vehicle declares no controller; apply one, such as NexusPx4API, to a Scope under this prim"
        )
    return controllers[0]


def declared_estimator(usd_path: str | pathlib.Path, registry: ComponentRegistry | None = None) -> ComponentSpec | None:
    """The estimator the vehicle file at `usd_path` declares, on the `Scope` that applies its schema, or ``None``.

    With no `registry`, the default one resolves each schema.

    Raises:
        ValueError: The vehicle declares more than one estimator, or one off a `Scope` whose parent is
            its root prim, its default prim; the message names their schemas and prims, or the prim off
            its place.
    """
    _, estimators = _on_scopes(usd_path, "estimator", registry)
    return estimators[0] if estimators else None


def prims_applying(usd_path: str | pathlib.Path, schema: str) -> list[tuple[str, list[str]]]:
    """Each active prim of the vehicle file that applies `schema`, as its path and every API schema it
    applies, so a caller can check what sits beside `schema`, as a peer's declaration beside its controller.
    """
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    return [
        (str(prim.GetPath()), list(applied))
        for prim in stage.Traverse(Usd.TraverseInstanceProxies())
        if schema in (applied := prim.GetAppliedSchemas())
    ]
