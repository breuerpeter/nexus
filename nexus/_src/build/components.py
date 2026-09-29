"""Resolve the components a vehicle Universal Scene Description (USD) file declares through nexus schemas."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

from nexus._src.core.registry import ComponentRegistry
from nexus._src.usd.reader import read_declarations


@dataclass(frozen=True)
class ComponentSpec:
    """One declared component: the prim that declares it, its schema, its class and its keyword arguments."""

    prim: str
    schema: str
    cls: type
    kwargs: dict


def resolve_components(usd_path: str | pathlib.Path, registry: ComponentRegistry | None = None) -> list[ComponentSpec]:
    """One record per applied nexus schema in the vehicle file at `usd_path`: its prim, class and keyword arguments.

    Creates no component.
    """
    return [
        ComponentSpec(prim, schema, registry.resolve(schema), kwargs)
        for prim, schema, kwargs in read_declarations(usd_path)
    ]
