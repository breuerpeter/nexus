"""Resolve the components a vehicle Universal Scene Description (USD) file declares through nexus schemas."""

from __future__ import annotations

import pathlib

from nexus._src.core.registry import ComponentRegistry


def resolve_components(usd_path: str | pathlib.Path, registry: ComponentRegistry | None = None) -> list:
    """One record per applied nexus schema in the vehicle file at `usd_path`: its prim, class and keyword arguments.

    Creates no component.
    """
    ...
