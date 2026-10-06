"""Scene ingestion: the model takes the resolved scene Universal Scene Description (USD) exactly as it
takes the vehicle USD.

The scene USD is the single authority on what it contributes: ``newton.ModelBuilder.add_usd``
imports exactly the ``UsdPhysics``-authored prims, for example the slalom pillars, whose authored
``physics:collisionEnabled = false`` makes them cost-only, and skips everything else: a
photogrammetry mesh or Gaussian-splat scan without physics schemas contributes nothing to the
model and lives purely in the render world. No scene names, kinds, or flags in code.
"""

import newton
import warp as wp


def add_scene(builder: newton.ModelBuilder, cfg: dict) -> list[int]:
    """Add the resolved scene USD, ``cfg['scene_usd_path']`` if any, to ``builder``.

    The registry scene's ``start``, ``cfg['scene_start']``, shifts the scene so the start point
    sits at the world origin: the same placement the render world gets. Returns the imported
    shape indices; a controller might need them, for example the obstacle cost of the sampling
    Model Predictive Control (MPC) controller.
    """
    path = cfg.get("scene_usd_path")
    if not path:
        return []
    sx, sy, sz = cfg.get("scene_start") or (0.0, 0.0, 0.0)
    before = builder.shape_count
    builder.add_usd(str(path), xform=wp.transform((-sx, -sy, -sz), wp.quat_identity()))
    return list(range(before, builder.shape_count))


__all__ = ["add_scene"]
