"""Scene ingestion: a scene is data, a catalog row plus one self-contained Universal Scene Description (USD) file.

:func:`add_scene` loads the scene into the physics model exactly as the vehicle USD loads: physics
takes the ``UsdPhysics``-authored prims, and everything else lives only in the render world. A scene
type that needs live machinery when it renders, the Cesium globe, gets it in the Kit render peer,
which claims the scene by the content of its USD, never by name. :mod:`.world` holds the world's
values the scene's ``PhysicsScene`` authors, as attributes of the model.
"""

from __future__ import annotations

from .ingest import add_scene
from .world import GRAVITY

__all__ = ["GRAVITY", "add_scene"]
