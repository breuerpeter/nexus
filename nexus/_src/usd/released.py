"""The check that a released schema version doesn't change in place."""

from __future__ import annotations

from pathlib import Path


def in_place_changes(plugin: Path) -> list[str]:
    """One message per released attribute the plugin folder at `plugin` lost, renamed, retyped or gave another unit.

    Each message names the schema version and the attribute. An attribute that joins a released version
    with a fallback is no change.
    """
    raise NotImplementedError
