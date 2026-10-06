Hosted assets are content-addressed on a flat scheme,
`<base>/assets/<kind>/<name>-<sha256>.<ext>`, where `<kind>` is `usd/vehicles` or `usd/scenes` for a
Universal Scene Description (USD) file and its `.glb` preview, and `policies` for an example's
exported `.pt` policy. `<name>` is the file's stem and `<base>` is what a catalog declares in its
`assets` block. The bucket behind the published base is a setting, `$NEXUS_BUCKET`, rather than a
line here, so no checkout carries an account's name.

- CloudFront serves the shipped catalog's base anonymously. A catalog that keeps blobs nobody else
  can read names them by full `s3://` URL, and the `aws` command-line tool fetches those with
  Amazon Web Services (AWS) credentials. CI artifacts, the docs `.rrd` embeds and the bench feed,
  live on the same bucket under `public/ci/{logs,bench}/`.
- The Kit-only scripts, `obj_to_usd.py`, `site_scan_splat.py` and `author_cesium_scene.py`, run
  from the host with `uv run python scripts/assets/<script>.py …`. Each starts the Kit image with
  its own file through `kit_container.py` and boots Kit there. The boot call sits in the script
  itself, and `tests/peers/kit/test_kit_single_door.py` allows it in these three files and the peer program
  only. A sibling module imports as `from scene_root import …`: the script's folder is on the path
  on both sides, and the working folder isn't.
- Uploads are **manual**. Prepare an asset with
  `uv run python scripts/assets/prepare_asset_upload.py {--vehicle|--scene} <usd> [--rotate-x 180]`.
  It sha256s the USD, prints where to upload it and the `nexus/_src/config/registry.yaml`
  snippet, and, for vehicles only, converts to a preview `.glb` in `assets/local/` via headless
  `bpy`. Pass `--rotate-x 180` for assets authored "up = -Z" such as Astro Max. Scenes get no
  `.glb`.
- A vehicle re-pin that changes a flown value changes the per-GPU trajectory hashes in
  `tests/build/test_stages.py`, a test that skips without CUDA. Record the local GPU's hash from
  the failing test, and the A10G's from the `gpu-pytest` log of a pull request that carries the
  `gpu` label.
