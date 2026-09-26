"""Scene ingestion: a scene is data, a registry row plus one self-contained Universal Scene Description (USD) file.

:func:`add_scene` loads the scene into the physics model exactly as the vehicle USD loads: physics
takes the ``UsdPhysics``-authored prims, and everything else lives only in the render world. A scene
type that needs live machinery when it renders, the Cesium globe, gets it in the Kit render peer,
which claims the scene by the content of its USD, never by name.
"""

from __future__ import annotations

from .ingest import add_scene

__all__ = ["add_scene"]
