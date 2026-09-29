"""Minimal plugin registry: the Scenario binds each interface to a provider, as
architecture.md §7 describes. Full entry-point discovery across installed packages is the
forward design; the slice registers providers explicitly, newton-cli wires them,
so the binding indirection exists without the packaging machinery yet.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from importlib.metadata import EntryPoint, entry_points

from nexus._src.usd import ENTRY_POINT_GROUP


class Registry:
    def __init__(self):
        self._providers: dict[str, dict[str, Callable]] = {}

    def register(self, interface: str, name: str, factory: Callable) -> None:
        self._providers.setdefault(interface, {})[name] = factory

    def resolve(self, interface: str, name: str) -> Callable:
        try:
            return self._providers[interface][name]
        except KeyError as e:
            available = list(self._providers.get(interface, {}))
            raise KeyError(f"No provider '{name}' for interface '{interface}'. Available: {available}") from e


class ComponentRegistry:
    """A value that maps each applied schema to the class that builds it, by class or import path."""

    def __init__(self, entries: dict[str, type | str | EntryPoint] | None = None):
        self._entries = dict(entries or {})

    def resolve(self, schema: str) -> type | None:
        """The class `schema` maps to, or `None` when no entry claims it.

        An entry given as an import path, `module:Class`, imports its module here, not before.
        """
        target = self._entries.get(schema)
        if isinstance(target, str):
            target = EntryPoint(name=schema, value=target, group=ENTRY_POINT_GROUP)
        return target.load() if isinstance(target, EntryPoint) else target


@cache
def default_registry() -> ComponentRegistry:
    """The default registry: every entry in the `nexus.components` entry-point group, read on first use."""
    return ComponentRegistry({entry.name: entry for entry in entry_points(group=ENTRY_POINT_GROUP)})


def register_component(schema: str, target: type | str) -> None:
    """Add an entry to the default registry."""
    ...
