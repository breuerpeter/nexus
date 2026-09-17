---
description: "Convert a photogrammetry mesh or Gaussian splat into a flyable visual world: container conversion, start pick, preview flight, and publishing to the registry."
---

# Converting scenes

Turn a captured 3D scan, an Esri Site Scan **`.obj`** export or a 3D-Tiles **Gaussian splat**
export, into a registry [scene](../reference/assets/scenes.md): a visual world the First Person
View (FPV) vehicles fly in. The output is always one self-contained `.usdz`, finalized onto the
scene-asset conventions in `scripts/assets/scene_root.py`. That means one root `Xform`, the
`defaultPrim` and the target the runtime places `start` at, holding everything, including the
authored **sky** of dome plus sun. There is no in-code sky. The scene file, a
Universal Scene Description (USD), owns its lighting. See
[what every scene file carries](../reference/assets/scenes.md#what-every-scene-file-carries).

Keep the scans under **`NEXUS_DATA`**, default `~/data`. That's the directory the Isaac Sim
container mounts at the *same* path it has on the host. So a path you type in step 1 is the path
step 3 flies, with no per-command `-v`, and the same command works from either side.

## 1. Convert

Both converters need a booted Kit app, which `nexus script` provides: it runs them inside the
`isaacsim` container, launching it if it isn't up. Progress traces on **stderr**. Kit swallows
stdout.

```bash
# photogrammetry OBJ (a dir with .obj + .mtl + textures) -> one .usdz
uv run nexus script scripts/assets/obj_to_usd.py \
    "$NEXUS_DATA/scans/obj/My Site" --out "$NEXUS_DATA/scans/my_site_obj.usdz"

# 3D-Tiles Gaussian splat (dir with tileset.json) -> one .usdz
uv run nexus script scripts/assets/site_scan_splat.py \
    "$NEXUS_DATA/scans/gsp/My Site" --out "$NEXUS_DATA/scans/my_site_splat.usdz" \
    --max-geometric-error 2.5
```

`--max-geometric-error` picks the splat Level Of Detail (LOD), where `0` is the finest level, the
leaves. The splat converter traces the latitude, longitude, and altitude of the **scene origin**.
Keep it for step 2. Conversion normalizes the scene, to Z-up meters with the XY origin at the
center of the scan's bounding box, but bakes in no start point.

## 2. Pick the start on the host

```bash
uv run python scripts/assets/spawn_site.py "$NEXUS_DATA/scans/my_site_splat.usdz" \
    --geo <lat>,<lon>,<alt>   # the converter's "scene origin" trace
```

Prints the registry `start:`, a flat open-terrain spot near the scene center, and, with `--geo`,
the `geodetic_origin` measured **at that start**. Both are advisory registry data: eyeball, tweak
numbers, no reconversion. A mesh carries no geodesy, so reuse the sibling splat's origin trace and
omit `alt`.

## 3. Preview-fly it unregistered

```bash
uv run nexus run --runtime isaacsim --vehicle astro_max_fpv --control px4-sitl \
    --scene "$NEXUS_DATA/scans/my_site_splat.usdz" --log
```

A local `--scene` path flies the file in place, from the scene's own origin. Inspect the
`.rrd` in Rerun, and check `fpvcam`.

## 4. Publish

```bash
uv run python scripts/assets/prepare_asset_upload.py --scene "$NEXUS_DATA/scans/my_site_splat.usdz"
```

Prints the content-addressed upload key, `public/assets/usd/scenes/<name>-<sha256>.usdz`, and the
`registry.yaml` snippet. Upload it by hand, then add the scene row with `start` and
`geodetic_origin` from step 2. Done: `--scene my_site_splat` flies it anywhere.
