"""Resolve the components a vehicle Universal Scene Description (USD) file declares through nexus schemas."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

from nexus_sim._src.usd.reader import ROLES, read_declarations, roles

from .registry import ComponentRegistry, default_registry


@dataclass(frozen=True)
class ComponentSpec:
    """One declared component: the prim that declares it, its schema, its class, its keyword arguments and the
    role its schema states.
    """

    prim: str
    schema: str
    cls: type
    kwargs: dict
    role: str


def resolve_components(usd_path: str | pathlib.Path, registry: ComponentRegistry | None = None) -> list[ComponentSpec]:
    """One record per applied nexus schema in the vehicle file at `usd_path`: its prim, class, keyword arguments
    and role.

    Creates no component. With no `registry`, the default one resolves each schema.

    Raises:
        ValueError: A prim applies a schema that states no role or more than one, or a schema no class
            claims; the message names the prim and the schema, and the roles it states.
    """
    if registry is None:
        registry = default_registry()
    specs = []
    for prim, schema, kwargs in read_declarations(usd_path):
        stated = roles(schema)
        if not stated:
            raise ValueError(
                f"{prim}: {schema} states no role; a component schema includes one role schema as a built-in, "
                f"one of {', '.join(ROLES)}"
            )
        if len(stated) > 1:
            raise ValueError(
                f"{prim}: {schema} states {len(stated)} roles, {' and '.join(stated)}; a component schema states one"
            )
        cls = registry.resolve(schema)
        if cls is None:
            raise ValueError(
                f"{prim}: no class claims {schema}; map it with register_component or a nexus.components entry point"
            )
        specs.append(ComponentSpec(prim, schema, cls, kwargs, stated[0]))
    return specs
