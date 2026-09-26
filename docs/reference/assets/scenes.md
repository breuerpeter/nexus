---
description: "The static worlds nexus flies in: flat ground, the slalom pillars, and the Cesium streamed globe. Selected by name with --scene."
---

# Scenes

A scene is the **static world** the vehicle flies in. Pick one by name with
`nexus run --scene <name>`, default `empty`. The registry,
`nexus/_src/config/registry.yaml`, resolves scenes. A local `.usdz` path such as
`--scene /path/to/scene.usdz` flies that file directly, which is the scene-development workflow.
See [Converting scenes](../../guide/convert-scenes.md).

**A scene is data: one self-contained Universal Scene Description (USD) file, ingested the same
way as the vehicle USD.** The physics adds it to the Newton `ModelBuilder`. When the vehicle
renders, the Kit render peer opens it as its stage, alongside the vehicle. The physics engine takes
exactly each prim that authors a `UsdPhysics` schema, such as the slalom pillars, whose authored
`physics:collisionEnabled = false` makes them cost-only. The renderer shows everything. The USD is
the single authority on what it contributes, with no scene kinds, flags, or names in code. A scene
type whose world can't live in a stage prim gets live machinery in the Kit peer's program, which
claims the scene by its USD content. The cesium globe's token injection and tile streaming live
there, in `nexus/_src/rendering/kit-peer/cesium_globe.py`.

Any scene flies with any vehicle. Whether it shows depends on the vehicle: a vehicle whose USD
authors RTX sensors renders them in the Kit peer, and there the scene is visible. A vehicle with none
simulates every physics prim of the same scene without a render.

| Scene | Description |
|---|---|
| `empty` | Flat ground, no static geometry, the default. A sky-only USD, because every scene carries its own lighting. |
| `slalom` | Three obstacle pillars for the sampling Model Predictive Control (MPC) demo, cost-only by authored `physics:collisionEnabled = false`. The Signed Distance Field (SDF) avoidance cost includes them, and the solver never contacts them. Avoidance is entirely the controller's job. |
| `cesium` | A live-streamed globe of Google Photorealistic 3D Tiles through Cesium for Omniverse. Anchored at the registry default origin, Seattle. Fly it over any location with `--geo <lat>,<lon>`, for example San Francisco with `--geo 37.7942,-122.3954`. Streams only where there is a renderer. Needs `CESIUM_ION_TOKEN`, see [Running](../../guide/running.md). |

## What every scene file carries

The converters and authoring scripts, in `scripts/assets/scene_root.py`, normalize scene assets
onto three conventions:

- **One root prim**: a plain `Xform`, the `defaultPrim` with no transform ops, holding everything.
  The renderer opens the scene USD as its **root stage** and places the registry `start` with a
  single translate on this prim, authored on the *session layer*. That layer is runtime-only, so
  the asset never changes. The vehicle composes in at the origin.
- **Its own sky**: a dome plus sun under the root. There is no in-code sky: a scene USD is the
  single authority on its own lighting.
- **Its render recipe, when it needs one**: stage render settings in the root layer's
  `customLayerData.renderSettings`, which Kit auto-applies when the stage opens. The cesium scene
  uses one for its fog, tone-map, and bloom grade. No applier code.

The cesium scene is exempt from the root-prim rule: its geographic reference anchors it, with no
`start`, and each `/Cesium*` prim lives at an authored absolute path.

## Placement: `start` and `geodetic_origin`

The converters normalize scene assets to Z-up meters with the XY origin at the center of the
scan's bounding box. *The asset doesn't bake in where the drone starts.* Two registry fields place
a scene:

- `start: [x, y, z]`: the scene-frame point the runtime moves to the world origin, so the
  drone, placed at the origin on the z=0 physics ground, starts on that surface.
  `scripts/assets/spawn_site.py` suggests one. Tweaking it takes a registry edit, no reconversion.
- `geodetic_origin: { lat, lon[, alt] }`: where the **world origin**, which is the start point,
  sits on Earth. It initializes the Hardware In The Loop (HIL) Global Positioning System (GPS) and
  anchors a Cesium globe. `alt` is the WGS84 *ellipsoidal* height of the start surface. It's
  optional and serves GPS realism only. The cesium scene measures it from the streamed tiles
  instead.
