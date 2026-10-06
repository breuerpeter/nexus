"""Resolve the components a vehicle Universal Scene Description (USD) file declares through nexus schemas."""

from __future__ import annotations

import pathlib

from nexus_sim._src.core.components import ComponentSpec, resolve_components
from nexus_sim._src.core.registry import ComponentRegistry

__all__ = ["ComponentSpec", "declared_controller", "resolve_components", "root_schemas"]


def _is_controller(cls: type) -> bool:
    """Whether `cls` implements the controller seam: `connect`, `close` and `stages`."""
    return all(callable(getattr(cls, name, None)) for name in ("connect", "close", "stages"))


def declared_controller(usd_path: str | pathlib.Path, registry: ComponentRegistry | None = None) -> ComponentSpec:
    """The one controller the vehicle file at `usd_path` declares on its root prim, its default prim.

    With no `registry`, the default one resolves each schema.

    Raises:
        ValueError: The vehicle declares no controller, more than one, or one off its root prim; the
            message names the prim, and for two or more, their schemas.
    """
    from pxr import Usd

    root = str(Usd.Stage.Open(str(usd_path)).GetDefaultPrim().GetPath()) or str(usd_path)
    controllers = [spec for spec in resolve_components(usd_path, registry) if _is_controller(spec.cls)]
    for spec in controllers:
        if spec.prim != root:
            raise ValueError(f"{spec.prim}: {spec.schema} declares a controller off the vehicle's root prim {root}")
    if not controllers:
        raise ValueError(f"{root}: the vehicle declares no controller; apply one, such as NexusPx4API, to this prim")
    if len(controllers) > 1:
        schemas = ", ".join(spec.schema for spec in controllers)
        raise ValueError(f"{root}: the vehicle declares {len(controllers)} controllers, {schemas}; it flies one")
    return controllers[0]


def root_schemas(usd_path: str | pathlib.Path) -> tuple[str, list[str]]:
    """The vehicle file's root prim, its default prim, and the API schemas applied to it, which a peer's
    declaration is one of.
    """
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path))
    root = stage.GetDefaultPrim()
    return str(root.GetPath()), list(root.GetAppliedSchemas())
