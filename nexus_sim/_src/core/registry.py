"""The registry: the class each name a run resolves maps to, a value the builder takes.

A name is an applied schema, a component's or a declared peer's, or the name a class requires a peer by. The
install answers every name a registry doesn't map itself: the `nexus.registry` entry-point group of each
installed package, one entry per schema, which :func:`register` adds to for a class defined in a script. A
test hands the run a registry of its own, with its stand-ins and its fakes as rows, so no test changes the
install's table.
"""

from __future__ import annotations

from functools import cache
from importlib.metadata import EntryPoint, entry_points

from nexus_sim._src.usd import ENTRY_POINT_GROUP

Target = type | str | EntryPoint


class Registry:
    """The classes a run resolves its names to: this registry's own rows, then the install's.

    Args:
        entries: The rows of this registry: each name to a class or an import path, `module:Class`. A row here
            answers before the install does, so a test maps a schema to a stand-in and the install answers
            the rest. ``Registry()`` is the install alone.
    """

    def __init__(self, entries: dict[str, type | str] | None = None):
        self._entries: dict[str, Target] = dict(entries or {})

    def add(self, name: str, target: type | str) -> None:
        """Map `name` to `target`, a class or an import path, `module:Class`, replacing any row it had."""
        self._entries[name] = target

    def resolve(self, name: str) -> type | None:
        """The class `name` maps to: this registry's row, else the install's, or ``None`` when neither claims it.

        An import path, or the install's entry point, imports its module here, not before, so a class with heavy
        imports costs nothing until a run declares its schema.
        """
        target = self._entries.get(name)
        if target is None:
            target = _installed().get(name)
        if target is None:
            return None
        if isinstance(target, str):
            target = EntryPoint(name=name, value=target, group=ENTRY_POINT_GROUP)
        return target.load() if isinstance(target, EntryPoint) else target


@cache
def _installed() -> dict[str, Target]:
    """The install's table: every entry in the `nexus.registry` group, read on first use, and what `register` adds."""
    return {entry.name: entry for entry in entry_points(group=ENTRY_POINT_GROUP)}


def register(name: str, target: type | str) -> None:
    """Map `name` to `target`, a class or an import path, `module:Class`, in the install's table, for a class a
    script defines rather than installs.

    Every registry that maps `name` itself keeps its own row.
    """
    _installed()[name] = target
