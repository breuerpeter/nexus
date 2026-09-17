"""Shared scene-asset finalization: the one root prim + the authored sky, the stage-5 conventions.

Every generic, non-cesium, scene asset ships as one self-contained ``.usdz`` whose root layer obeys:

* **One root prim**: a plain ``Xform`` with no xformOps, set as the ``defaultPrim``, holding
  everything: geometry / splat / obstacle prims / lights. The runtime opens the scene Universal
  Scene Description (USD) as the root stage and applies the registry ``start`` as a single
  ``-start`` translate on this prim on the session layer, runtime-only, never dirtying the asset,
  while the vehicle composes in at the origin, the drone-spawns-at-origin convention.
* **The sky lives in the asset**: a dome + sun authored under the root prim. The runtime's old
  ``_add_sky`` no longer exists; a scene USD is the single authority on its own lighting.
  Deliberately not a photo High Dynamic Range Image (HDRI): a captured-scene HDRI reads as "the
  environment" and masks the actual scene.

The cesium scene is exempt: it's handler-claimed, anchored by its georeference with no ``start``,
and its ``/Cesium*`` prims live at authored absolute paths.

``pxr``-only: runs host-side or in the Kit container.
"""

from __future__ import annotations

# The plain lit sky, dome + sun: the values the runtime's _add_sky used to author per run.
SKY_DOME_INTENSITY = 1200.0
SKY_DOME_COLOR = (0.55, 0.68, 0.92)
SKY_SUN_INTENSITY = 2200.0
SKY_SUN_ANGLE = 0.5
SKY_SUN_ROTATE = (-50.0, 0.0, 35.0)


def author_sky(stage, root_path: str, *, dome_intensity: float = SKY_DOME_INTENSITY) -> None:
    """Author the plain lit sky, dome + sun and no geometry, under ``root_path``.

    Image-based lighting for the vehicle and the scene geometry; riding under the scene root means
    the runtime's ``-start`` translate moves the lights with the scene, which is harmless: a dome
    light is positionless and a distant light only has direction.
    """
    from pxr import Gf, UsdGeom, UsdLux

    dome = UsdLux.DomeLight.Define(stage, f"{root_path}/EnvSky")
    dome.CreateIntensityAttr(float(dome_intensity))
    dome.CreateColorAttr(Gf.Vec3f(*SKY_DOME_COLOR))
    UsdGeom.Xformable(dome.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, 0.0))  # horizon level, Z-up
    sun = UsdLux.DistantLight.Define(stage, f"{root_path}/EnvSun")
    sun.CreateIntensityAttr(SKY_SUN_INTENSITY)
    sun.CreateAngleAttr(SKY_SUN_ANGLE)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(*SKY_SUN_ROTATE))


def finalize_scene_layer(usd_path: str, *, root_name: str = "World", sky: bool = True) -> str:
    """Normalize a converted/authored scene layer onto the stage-5 conventions, in place.

    Ensures exactly one root prim: an existing single root ``Xform`` stays, provided its xformOps
    are clear, since the runtime's session translate must be the only opinion; any other shape,
    whether more than one root or a typed root such as a splat's ``ParticleField3DGaussianSplat``,
    moves under a fresh ``/<root_name>`` Xform via a namespace edit. Sets ``defaultPrim`` to the
    root and authors the sky beneath it. Returns the root prim path.
    """
    from pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError(f"could not open USD stage: {usd_path}")
    layer = stage.GetRootLayer()

    roots = list(stage.GetPseudoRoot().GetChildren())
    clean_single_root = (
        len(roots) == 1 and roots[0].IsA(UsdGeom.Xform) and not UsdGeom.Xformable(roots[0]).GetOrderedXformOps()
    )
    if clean_single_root:
        root_path = str(roots[0].GetPath())
    else:
        # Re-root: move every existing root under a fresh, transform-free Xform via a namespace edit
        # on the layer. Covers a typed root such as a splat's ParticleField, more than one root, and
        # a root that authors xformOps: the asset converter bakes its up-axis correction on /World,
        # and those ops must compose beneath the runtime's session start-translate, not fight it.
        taken = {str(p.GetName()) for p in roots}
        root_path = "/" + next(n for n in (root_name, "Scene", "SceneRoot") if n not in taken)
        stage.DefinePrim(root_path, "Xform")
        edit = Sdf.BatchNamespaceEdit()
        for p in roots:
            edit.Add(Sdf.NamespaceEdit.Reparent(p.GetPath(), Sdf.Path(root_path), -1))
        if not layer.Apply(edit):
            raise RuntimeError(f"namespace re-root failed for {usd_path}")

    stage.SetDefaultPrim(stage.GetPrimAtPath(root_path))
    if sky:
        author_sky(stage, root_path)
    layer.Save()
    return root_path
