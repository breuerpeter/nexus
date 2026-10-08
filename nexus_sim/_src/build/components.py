"""Resolve the components a vehicle Universal Scene Description (USD) file declares through nexus schemas."""

from __future__ import annotations

import pathlib

from nexus_sim._src.core.components import ComponentSpec, resolve_components
from nexus_sim._src.core.registry import ComponentRegistry

__all__ = ["ComponentSpec", "declared_controller", "prims_applying", "resolve_components"]


def _is_controller(cls: type) -> bool:
    """Whether `cls` implements the controller seam: `connect`, `close` and `stages`."""
    return all(callable(getattr(cls, name, None)) for name in ("connect", "close", "stages"))


def declared_controller(usd_path: str | pathlib.Path, registry: ComponentRegistry | None = None) -> ComponentSpec:
    """The one controller the vehicle file at `usd_path` declares, on the `Scope` that applies its schema.

    With no `registry`, the default one resolves each schema.

    Raises:
        ValueError: The vehicle declares no controller, more than one, or one outside its root prim,
            its default prim; the message names the root prim, and for two or more, their schemas and
            prims, or the prim outside the root.
    """
    from pxr import Sdf, Usd

    default = Usd.Stage.Open(str(usd_path)).GetDefaultPrim().GetPath()  # empty with no default prim
    root = str(default) or str(usd_path)
    controllers = [spec for spec in resolve_components(usd_path, registry) if _is_controller(spec.cls)]
    for spec in controllers:
        if not default.isEmpty and not Sdf.Path(spec.prim).HasPrefix(default):
            raise ValueError(
                f"{spec.prim}: {spec.schema} declares a controller outside the vehicle's root prim {root}; "
                "put its Scope under the root prim"
            )
    if not controllers:
        raise ValueError(
            f"{root}: the vehicle declares no controller; apply one, such as NexusPx4API, to a Scope under this prim"
        )
    if len(controllers) > 1:
        placed = ", ".join(f"{spec.schema} on {spec.prim}" for spec in controllers)
        raise ValueError(f"{root}: the vehicle declares {len(controllers)} controllers, {placed}; it flies one")
    return controllers[0]


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
