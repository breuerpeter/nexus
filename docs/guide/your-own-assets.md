---
description: "Fly your own vehicles and scenes: what the shipped catalog holds, how a run finds a catalog, contributing an asset, and keeping a catalog and a host of your own."
---

# Bringing your own assets

The framework installs with one catalog: the [Astro Max variants](../reference/assets/vehicles.md) and the
[scenes](../reference/assets/scenes.md) the project hosts and serves. That catalog is a starting
point, not the boundary. A project of your own flies your vehicles in your scenes by keeping a
catalog beside it, with no fork of this one and no patch on top of it.

## How a run finds its catalog

A run loads its catalogs in layers, each over the ones before it:

1. the catalog bundled in the package
2. the catalog of each installed package that adds one, see
   [A package that adds a catalog](#a-package-that-adds-a-catalog)
3. a project catalog of your own, the first hit of:
    1. the path a run names: `nexus run --vehicle <name> --scene <name> --catalog <path>`, or
       `Sim("<name>", scene="<name>", catalog=<path>)` in a script
    2. the nearest **`nexus.catalog.yaml`** in the working directory or a directory over it, the same
       walk up that `uv`, `ruff` and `pytest` do for their own files

Each catalog extends the layers below it, so it lists only what it adds, and a run can still fly every
bundled vehicle and scene. A name that two layers define takes the upper layer's entry, and the load logs a
warning with the name and both hashes. That's how you pin a bundled asset to another version on
purpose, and how a stale copy of an entry shows up. Two installed packages that define one name fail
the load, since no order between them means anything. A catalog
holds no defaults: every run names its vehicle and its scene, `nexus run --vehicle my_quad --scene empty` or
`Sim("my_quad", scene="empty")`, and a catalog that still carries a `defaults` block fails to load.
Every run names the catalogs it loaded in its first log line. Its tested-config receipt lists them, in load order, under
`catalogs`: each one's kind, `bundled`, `package` or `project`, its owner, such as the package's distribution name, and its file. A recording then says whose entries produced it.

## A package that adds a catalog

A package installed beside the framework adds its catalog on install, the way it adds its
[components](your-own-component.md#the-registration). One entry in the `nexus.catalog` group of its
`pyproject.toml` names the package, and the framework reads `nexus.catalog.yaml` from that package's
folder:

```toml
[project.entry-points."nexus.catalog"]
my_drones = "my_drones"
```

```text
my_drones/
├── __init__.py
├── nexus.catalog.yaml     # the vehicles and scenes the package adds
└── nexus.px4_sitl.ref     # optional: the PX4 tree its vehicles fly
```

Once installed, a script names a vehicle of the package from any directory, with no catalog path. An
entry point whose package holds no `nexus.catalog.yaml` fails the load. The package ships the catalog
file in its wheel as package data. A pin file beside it, such as `nexus.px4_sitl.ref`, pins a peer as a
project's does, and a project's own pin file overrides it, see [PX4 as a peer](running.md#px4-as-a-peer).
Two installed packages that pin one peer fail the run unless a project's file or the peer's variable
picks the pin.

## What a catalog looks like

```yaml
assets:
  base: https://assets.example.com/catalog   # where this catalog's blobs live

vehicles:
  my_quad:
    usd: { name: my_quad, sha256: 3f1c…d92 }             # completed against the base
  borrowed:
    usd: { url: "https://cdn.example/x-3f1c…d92.usdz", sha256: 3f1c…d92 }   # hosted elsewhere
  in_progress:
    usd: { url: "file:///home/me/assets/draft.usdz", sha256: 9ab2…40f }     # still being authored
```

A vehicle entry names no controller: the vehicle's Universal Scene Description (USD) file declares
it. PX4 flies a vehicle whose controller's scope applies `NexusPx4API_1`. The scope of its PX4 Software
In The Loop peer applies `NexusPx4SitlAPI` and names the airframe in `nexus:airframe`, the make target
without PX4's `none_` prefix. Each scope is a `Scope` prim under the vehicle's root prim, named
`Controller` and `Px4Sitl` by convention. Each rotor
declares its propeller with `NexusPropellerAPI` on its rigid body, and its motor with a
`NewtonActuator` prim that drives its joint, whose `newton:velocityLimit` is the rotor speed at full
command. The [schema reference](../reference/schemas.md) lists the attributes.

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
Universal Scene Description (USD) file, prints the key to put it at and the catalog snippet to
paste, and for a vehicle converts a `.glb` preview.

An asset is one self-contained `.usdz`, one blob and one sha256, so package a multi-file USD before
publishing it. [Converting scenes](convert-scenes.md) covers turning a scan into one.

## Contributing an asset to this catalog

Vehicles and scenes anyone can fly are welcome in the shipped catalog. Two steps, in this order:

1. **The blob goes up first.** Open an issue with the asset and a maintainer publishes it. A key
   holds the content's own hash, so publishing early is harmless: nothing can overwrite it and
   nothing points at it yet.
2. **Then the entry merges.** A pull request adds the `{name, sha256}` entry to the catalog. The
   other order leaves the catalog naming a blob nobody can fetch, which fails every run that names
   it.
