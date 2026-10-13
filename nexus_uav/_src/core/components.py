"""Resolve the components and the peers a vehicle Universal Scene Description (USD) file declares through nexus
schemas.

The registry maps each applied schema to its class. A component schema states its role by including a role
schema as a built-in. A peer schema states no role: the class it maps to is a peer, one with ``start``,
``stop`` and ``alive``, and so the build sorts a declared peer from a component.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

from nexus_uav._src.usd.reader import ROLES, read_declarations, roles

from .registry import Registry


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


@dataclass(frozen=True)
class PeerSpec:
    """One declared peer: the prim that declares it, its schema, its class and its keyword arguments."""

    prim: str
    schema: str
    cls: type
    kwargs: dict


def is_peer(cls: type) -> bool:
    """Whether `cls` is a peer's class: it states ``start``, ``stop`` and ``alive``."""
    return all(callable(getattr(cls, name, None)) for name in ("start", "stop", "alive"))


def resolve_declarations(
    usd_path: str | pathlib.Path, registry: Registry | None = None
) -> tuple[list[ComponentSpec], list[PeerSpec]]:
    """One record per applied nexus schema in the vehicle file at `usd_path`, a component's or a peer's, in the
    order of the prims.

    Creates nothing. With no `registry`, the install resolves each schema.

    Raises:
        ValueError: A prim applies a schema no class claims, a component schema that states no role or more
            than one, or a peer schema that states a role; the message names the prim and the schema, and the
            roles it states.
    """
    if registry is None:
        registry = Registry()
    components, peers = [], []
    for prim, schema, kwargs in read_declarations(usd_path):
        cls = registry.resolve(schema)
        if cls is None:
            raise ValueError(f"{prim}: no class claims {schema}; map it with register or a nexus.registry entry point")
        stated = roles(schema)
        if is_peer(cls):
            if stated:
                raise ValueError(f"{prim}: {schema} declares a peer, {cls.__name__}, and states the {stated[0]} role")
            peers.append(PeerSpec(prim, schema, cls, kwargs))
            continue
        if not stated:
            raise ValueError(
                f"{prim}: {schema} states no role; a component schema includes one role schema as a built-in, "
                f"one of {', '.join(ROLES)}"
            )
        if len(stated) > 1:
            raise ValueError(
                f"{prim}: {schema} states {len(stated)} roles, {' and '.join(stated)}; a component schema states one"
            )
        components.append(ComponentSpec(prim, schema, cls, kwargs, stated[0]))
    return components, peers


def resolve_components(usd_path: str | pathlib.Path, registry: Registry | None = None) -> list[ComponentSpec]:
    """One record per component the vehicle file at `usd_path` declares: its prim, class, keyword arguments and
    role. :func:`resolve_declarations` states what fails.
    """
    return resolve_declarations(usd_path, registry)[0]
