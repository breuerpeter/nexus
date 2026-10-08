"""Resolve a ``LaunchConfig`` against the catalog into a ``ResolvedLaunch``.

Flow: look the named vehicle variant up → resolve the named scene → fetch and sha-verify every ``{url, sha256}`` asset, for the vehicle and scene
Universal Scene Description (USD) files and the policy → emit the tested-config receipt plus local
paths.
"""

from __future__ import annotations

import hashlib
import pathlib

from nexus_sim._src.core import logger

from .catalog import Catalog, CatalogError, NoMatchError, VehicleVariant, catalog_path, load_catalog
from .models import AssetRef, LaunchConfig
from .receipt import ResolvedLaunch, TestedConfig


def _resolve_asset(ref, cache_dir):
    if ref is None:
        return None
    try:
        from nexus_sim._src.assets import resolve as _fetch
    except ImportError:  # pragma: no cover, depends on how nexus_sim._src.assets re-exports
        from nexus_sim._src.assets.resolver import resolve as _fetch
    kw = {"cache_dir": cache_dir} if (cache_dir is not None and cache_dir != "") else {}
    return _fetch(ref.model_dump(exclude_none=True), **kw)


def _local_usd(name: str | None, kind: str) -> pathlib.Path | None:
    """A vehicle or scene name that's really a local USD path: ``--vehicle path/to/model.usdz`` or
    ``--scene path/to/scene.usd``.
    """
    if not name:
        return None
    if not (name.endswith((".usd", ".usda", ".usdc", ".usdz")) or "/" in name):
        return None
    path = pathlib.Path(name).expanduser()
    if not path.exists():
        raise CatalogError(f"local {kind} USD not found: {path}")
    return path.resolve()


def _local_ref(path: pathlib.Path) -> AssetRef:
    """The receipt entry for a local file: its URL, its sha256 and its name, the form a hosted asset takes."""
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return AssetRef(url=path.as_uri(), sha256=sha, filename=path.name)


def _layer(path: str | None) -> tuple[pathlib.Path, AssetRef] | None:
    """The override layer a launch names, as its local path and its receipt entry, or ``None``.

    Raises:
        FileNotFoundError: No file is at the path; the message names it.
    """
    if path is None:
        return None
    local = pathlib.Path(path).expanduser()
    if not local.is_file():
        raise FileNotFoundError(f"override layer not found: {local}")
    local = local.resolve()
    return local, _local_ref(local)


def _stack(layer: pathlib.Path, layer_sha: str, vehicle: pathlib.Path, vehicle_sha: str, cache_dir) -> pathlib.Path:
    """A root file in the asset cache that stacks *layer* over *vehicle*, so each reader of the vehicle
    opens the composed vehicle by one path. USD composes the two; the file only lists them, strongest
    first, and carries the vehicle's default prim.
    """
    from pxr import Sdf

    from nexus_sim._src.assets.resolver import default_cache

    out = pathlib.Path(cache_dir or default_cache()) / "layered" / f"{vehicle_sha}-{layer_sha}" / "vehicle.usda"
    out.parent.mkdir(parents=True, exist_ok=True)
    root = Sdf.Layer.CreateAnonymous(".usda")
    root.subLayerPaths = [str(layer), str(vehicle)]
    root.defaultPrim = Sdf.Layer.FindOrOpen(str(vehicle)).defaultPrim
    root.Export(str(out))
    return out


def resolve(
    launch: LaunchConfig,
    catalog: Catalog | None = None,
    *,
    fetch: bool = True,
    cache_dir: str | pathlib.Path | None = None,
) -> ResolvedLaunch:
    """Resolve *launch* against *catalog*, which defaults to the bundled one.

    With ``fetch=True``, the default, the resolver downloads and sha-verifies every ``{url, sha256}``
    asset into the local cache and returns its path: the vehicle USD and the scene USD.
    ``fetch=False`` resolves the receipt only, with no network, which is useful for tests and dry runs. ``cache_dir=None`` uses the default
    asset cache. The receipt carries the launch's device selector as given; the build stamps
    the device the run picks, since that needs Warp, which a caller that only wants an asset path
    never starts.
    """
    # A caller that hands over a Catalog owns it; otherwise the run finds its own catalog and says
    # which one it flew, so a recording beside it answers what produced it.
    source = None
    if catalog is None:
        source = catalog_path(launch.catalog)
        catalog = load_catalog(source)
        logger.info(f"catalog: {source}")

    if launch.vehicle is None:
        raise NoMatchError(f"the launch names no vehicle; name one of {list(catalog.vehicles)}")
    if launch.scene is None:
        raise CatalogError(f"the launch names no scene; name one of {list(catalog.scenes)}")
    local = _local_usd(launch.vehicle, "vehicle")
    if local is None:
        if launch.vehicle not in catalog.vehicles:  # `--vehicle <name>`
            raise NoMatchError(f"no vehicle named {launch.vehicle!r}; catalog names: {list(catalog.vehicles)}")
        variant = catalog.vehicles[launch.vehicle]
    else:
        # Local vehicle USD, the variant-development workflow: the file *is* the authority. Its
        # controller, actuator params, cameras, and lidars are all authored on it, so it needs no
        # catalog row. The receipt stays honest: the file gets a sha256 the same way as a catalog asset.
        variant = VehicleVariant(usd=_local_ref(local))

    scene_id = launch.scene
    local_scene = _local_usd(scene_id, "scene")
    if local_scene is not None:
        # Local scene USD, the scene-development workflow: a freshly converted mesh or splat, not yet
        # registered. What it *is* is the USD's own business: the run's builder routes by the physics
        # schemas the USD authors, or their absence. Used in place the same way as a local vehicle
        # USD, with no cache copy; the receipt stays honest, with a sha256 the same way as a catalog
        # asset. ``start`` is catalog data: a local scene flies from its own origin, and
        # scripts/assets/spawn_site.py helps pick one.
        from .catalog import Scene

        scene = Scene(usd=_local_ref(local_scene))
    elif scene_id not in catalog.scenes:
        raise CatalogError(f"unknown scene {scene_id!r}; have {list(catalog.scenes)}")
    else:
        scene = catalog.scenes[scene_id]

    layer = _layer(launch.layer)
    veh_path = local if local is not None else (_resolve_asset(variant.usd, cache_dir) if fetch else None)
    if layer is not None and veh_path is not None:
        veh_path = _stack(layer[0], layer[1].sha256, pathlib.Path(veh_path), variant.usd.sha256, cache_dir)
    if local_scene is not None:
        scn_path = local_scene  # in place: sibling files such as a mesh's textures/ must stay resolvable
    else:
        scn_path = _resolve_asset(scene.usd, cache_dir) if (fetch and scene.usd) else None
    # The geodetic origin: a launch override wins, else the catalog scene's default, which can be None.
    geodetic_origin = launch.geodetic_origin or scene.geodetic_origin
    tested = TestedConfig(
        vehicle=str(local) if local is not None else launch.vehicle,
        catalog=str(source) if source is not None else None,
        vehicle_usd=variant.usd,
        layer=layer[1] if layer is not None else None,
        scene=scene_id,
        scene_usd=scene.usd,
        scene_start=scene.start,
        geodetic_origin=geodetic_origin,
        runtime=launch.runtime,
    )
    return ResolvedLaunch(
        tested_config=tested,
        vehicle_usd_path=str(veh_path) if veh_path is not None else None,
        scene_usd_path=str(scn_path) if scn_path is not None else None,
    )
