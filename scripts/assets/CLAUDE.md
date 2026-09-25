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
- Uploads are **manual**. Prepare an asset with
  `uv run python scripts/assets/prepare_asset_upload.py {--vehicle|--scene} <usd> [--rotate-x 180]`.
  It sha256s the USD, prints where to upload it and the `nexus/_src/config/registry.yaml`
  snippet, and, for vehicles only, converts to a preview `.glb` in `assets/local/` via headless
  `bpy`. Pass `--rotate-x 180` for assets authored "up = -Z" such as Astro Max. Scenes get no
  `.glb`.
