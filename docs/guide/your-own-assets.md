---
description: "Fly your own vehicles and scenes: what the shipped catalog holds, how a run finds a catalog, contributing an asset, and keeping a registry and a host of your own."
---

# Bringing your own assets

nexus installs with one catalog: the [Astro Max variants](../reference/assets/vehicles.md) and the
[scenes](../reference/assets/scenes.md) the project hosts and serves. That catalog is a starting
point, not the boundary. A project of your own flies your vehicles in your scenes by keeping a
registry beside it, with no fork of this one and no patch on top of it.

## How a run finds its catalog

A run always loads the catalog bundled in the package. It then looks for a catalog of your own, in
order, first hit wins:

1. the path a run names: `nexus run --registry <path>`, or `Sim(registry=<path>)` in a script
2. the nearest **`nexus.registry.yaml`** in the working directory or a directory over it, the same
   walk up that `uv`, `ruff` and `pytest` do for their own files

Your catalog extends the bundled one, so it lists only what you add, and a run can still fly every
bundled vehicle and scene. A name that both catalogs define takes your entry, and the load logs a
warning with the name and both hashes. That's how you pin a bundled asset to another version on
purpose, and how a stale copy of a bundled entry shows up. Your `defaults` win key by key, so a
catalog that sets only `defaults.vehicle` still flies the bundled default scene. Every run names
the catalog it loaded in its first log line and records the path in its tested-config receipt, so a
recording says which catalog produced it.

## What a registry looks like

```yaml
assets:
  base: https://assets.example.com/catalog   # where this catalog's blobs live

vehicles:
  - name: my_quad
    usd: { name: my_quad, sha256: 3f1c…d92 }             # completed against the base
    px4: { airframe: my_quad }
  - name: borrowed
    usd: { url: "https://cdn.example/x-3f1c…d92.usdz", sha256: 3f1c…d92 }   # hosted elsewhere
  - name: in_progress
    usd: { url: "file:///home/me/assets/draft.usdz", sha256: 9ab2…40f }     # still being authored

defaults: { vehicle: my_quad }   # the scene stays the bundled default
```

An entry either carries a full `url` or a compact `{name, sha256}` that its own catalog's
`assets.base` completes into `<base>/assets/usd/{vehicles,scenes}/<name>-<sha256>.usdz`. The hosted
policy the `goto_policy` example flies sits beside them at `<base>/assets/policies/<name>-<sha256>.pt`.
The sha256 is the version pin: the resolver verifies it on fetch, and because it sits in the key the object
never changes and caches forever.

The scheme of the URL decides how a run reads it. `https://` is an anonymous GET with no
credentials. `file://` reads from disk. `s3://` shells out to the `aws` command-line tool, so the
ordinary credential chain applies: an instance profile, access keys or `aws sso login`. That's how
a catalog keeps blobs nobody else can read.

## Hosting your own

Any host that serves bytes over HTTPS works. This project uses a bucket behind a
Content Delivery Network (CDN), with objects at
`<prefix>/assets/usd/{vehicles,scenes}/<name>-<sha256>.usdz`. The sha256 sits in the key, so nothing
ever overwrites anything. `scripts/assets/prepare_asset_upload.py` prepares an asset: it hashes the
Universal Scene Description (USD) file, prints the key to put it at and the registry snippet to
paste, and for a vehicle converts a `.glb` preview.

An asset is one self-contained `.usdz`, one blob and one sha256, so package a multi-file USD before
publishing it. [Converting scenes](convert-scenes.md) covers turning a scan into one.

## Contributing an asset to this catalog

Vehicles and scenes anyone can fly are welcome in the shipped catalog. Two steps, in this order:

1. **The blob goes up first.** Open an issue with the asset and a maintainer publishes it. A key
   holds the content's own hash, so publishing early is harmless: nothing can overwrite it and
   nothing points at it yet.
2. **Then the entry merges.** A pull request adds the `{name, sha256}` entry to the registry. The
   other order leaves the catalog naming a blob nobody can fetch, which fails every run that names
   it.
