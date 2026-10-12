"""An override layer that moves a shipped scene's geodetic origin, for a flight whose plan is geodetic."""

from __future__ import annotations

import tempfile
from pathlib import Path

# Where each shipped scene's `PhysicsScene` prim sits, the prim that authors its origin.
PHYSICS_SCENE = "/World/PhysicsScene"


def origin_layer(lat: float, lon: float, alt: float) -> str:
    """Write an override layer that moves a shipped scene's origin to `lat`, `lon` and `alt`, and return its path.

    Pass the path as `Sim(layer=...)`. The layer authors `nexus:geodeticOrigin` on the scene's
    `PhysicsScene` prim: latitude and longitude in degrees, and the altitude over mean sea level in
    metres. It stays in the temporary folder, since the run reads it until it stops.
    """
    world, scene = PHYSICS_SCENE.strip("/").split("/")
    path = Path(tempfile.gettempdir()) / f"nexus-origin-{lat:.7f}-{lon:.7f}-{alt:.2f}.usda"
    path.write_text(
        f'#usda 1.0\n\nover "{world}"\n{{\n    over "{scene}"\n    {{\n'
        f"        double3 nexus:geodeticOrigin = ({lat!r}, {lon!r}, {alt!r})\n    }}\n}}\n"
    )
    return str(path)
