"""Resolve the components a vehicle Universal Scene Description (USD) file declares through nexus schemas."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

from nexus._src.usd.reader import read_declarations

from .registry import ComponentRegistry, default_registry


@dataclass(frozen=True)
class ComponentSpec:
    """One declared component: the prim that declares it, its schema, its class and its keyword arguments."""

    prim: str
    schema: str
    cls: type
    kwargs: dict


def resolve_components(usd_path: str | pathlib.Path, registry: ComponentRegistry | None = None) -> list[ComponentSpec]:
    """One record per applied nexus schema in the vehicle file at `usd_path`: its prim, class and keyword arguments.

    Creates no component. With no `registry`, the default one resolves each schema.

    Raises:
        ValueError: A prim applies a schema no class claims; the message names the prim and the schema.
    """
    if registry is None:
        registry = default_registry()
    specs = []
    for prim, schema, kwargs in read_declarations(usd_path):
        cls = registry.resolve(schema)
        if cls is None:
            raise ValueError(
                f"{prim}: no class claims {schema}; map it with register_component or a nexus.components entry point"
            )
        specs.append(ComponentSpec(prim, schema, cls, kwargs))
    return specs
