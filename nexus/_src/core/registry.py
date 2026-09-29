"""The component registry: the class each applied schema resolves to.

A registry is a value the builder takes. The default one reads the `nexus.components` entry-point group on
first use, and `register_component` adds to it. A test builds a registry of its own instead, so no test
changes module state.
"""

from __future__ import annotations

from functools import cache
from importlib.metadata import EntryPoint, entry_points

from nexus._src.usd import ENTRY_POINT_GROUP


class ComponentRegistry:
    """A value that maps each applied schema to the class that builds it, by class or import path."""

    def __init__(self, entries: dict[str, type | str | EntryPoint] | None = None):
        self._entries = dict(entries or {})

    def add(self, schema: str, target: type | str) -> None:
        """Map `schema` to `target`, a class or an import path, `module:Class`, replacing any entry it had."""
        self._entries[schema] = target

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
    """Map `schema` to `target`, a class or an import path, `module:Class`, in the default registry.

    A registry handed to the builder doesn't see it: a test builds its own registry and changes nothing
    here.
    """
    default_registry().add(schema, target)
