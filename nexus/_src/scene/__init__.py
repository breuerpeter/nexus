"""Per-scene-type code hooks: the one sanctioned home for scene machinery.

Scenes are data: a registry row with ``usd``, ``geodetic_origin`` and ``start``, plus one
self-contained Universal Scene Description (USD) file, loaded everywhere the same way as the
vehicle USD. :func:`add_scene` is that ingestion: physics takes the ``UsdPhysics``-authored prims,
the renderer shows everything. A scene type that needs live machinery beyond that ships a
:class:`SceneHandler` subclass here: the cesium globe needs a secret token injected, tile-selection
viewports, a streaming drain, and a ground-align probe. Selection is by USD **content** through
:meth:`SceneHandler.matches`, never by name: the scene USD stays the single authority.
"""

from __future__ import annotations

from .base import SceneHandler
from .cesium import CesiumGlobe
from .ingest import add_scene

_HANDLERS: tuple[type[SceneHandler], ...] = (CesiumGlobe,)


def handler_for(usd_path) -> SceneHandler | None:
    """The handler whose ``matches`` claims the scene USD, or ``None`` for the generic data-only path."""
    if not usd_path:
        return None
    for cls in _HANDLERS:
        if cls.matches(str(usd_path)):
            return cls()
    return None


__all__ = ["CesiumGlobe", "SceneHandler", "add_scene", "handler_for"]
