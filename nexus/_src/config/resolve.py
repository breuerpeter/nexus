"""Resolve a ``LaunchConfig`` against the registry into a ``ResolvedLaunch``.

Flow: look the named vehicle variant up, or take the registry's default → resolve the
scene → fetch and sha-verify every ``{url, sha256}`` asset, for the vehicle and scene
Universal Scene Description (USD) files and the policy → emit the tested-config receipt plus local
paths.
"""

from __future__ import annotations

import pathlib

from nexus._src.core import logger

from .models import LaunchConfig
from .receipt import ResolvedLaunch, TestedConfig
from .registry import NoMatchError, Registry, RegistryError, VehicleVariant, load_registry, registry_path


def _resolve_asset(ref, cache_dir):
    if ref is None:
        return None
    try:
        from nexus._src.assets import resolve as _fetch
    except ImportError:  # pragma: no cover, depends on how nexus._src.assets re-exports
        from nexus._src.assets.resolver import resolve as _fetch
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
        raise RegistryError(f"local {kind} USD not found: {path}")
    return path.resolve()


def resolve(
    launch: LaunchConfig,
    registry: Registry | None = None,
    *,
    fetch: bool = True,
    cache_dir: str | pathlib.Path | None = None,
) -> ResolvedLaunch:
    """Resolve *launch* against *registry*, which defaults to the bundled one.

    With ``fetch=True``, the default, the resolver downloads and sha-verifies every ``{url, sha256}``
    asset into the local cache and returns its path: the vehicle USD, the scene USD, and the policy
    weights when ``control.kind == 'policy'``. ``fetch=False`` resolves the receipt only, with no
    network, which is useful for tests and dry runs. ``cache_dir=None`` uses the default
    newton-assets cache.
    """
    # A caller that hands over a Registry owns it; otherwise the run finds its own catalog and says
    # which one it flew, so a recording beside it answers what produced it.
    source = None
    if registry is None:
        source = registry_path(launch.registry)
        registry = load_registry(source)
        logger.info(f"registry: {source}")

    local = _local_usd(launch.vehicle, "vehicle")
    if local is None:
        name = launch.vehicle if launch.vehicle is not None else registry.defaults.vehicle
        if name is None:
            raise NoMatchError(
                f"the launch names no vehicle and the registry has no default; "
                f"registry names: {[v.name for v in registry.vehicles]}"
            )
        variant = registry.by_name(name)  # `--vehicle <name>`, or the registry's default
    else:
        # Local vehicle USD, the variant-development workflow: the file *is* the authority. Actuator
        # params, cameras, and lidars are all authored on it, so it needs no registry row. The receipt
        # stays honest: the file gets a sha256 the same way as a registry asset; the PX4 spec falls
        # back to the registry-default variant's, receipt-only, since whoever runs PX4 picks the
        # Software In The Loop (SITL) airframe.
        import hashlib

        from .models import AssetRef

        sha = hashlib.sha256(local.read_bytes()).hexdigest()
        default = registry.by_name(registry.defaults.vehicle) if registry.defaults.vehicle else None
        variant = VehicleVariant(
            name=str(local),
            usd=AssetRef(url=local.as_uri(), sha256=sha, filename=local.name),
            px4=default.px4 if default is not None else None,
        )

    scene_id = launch.scene or registry.defaults.scene
    local_scene = _local_usd(scene_id, "scene")
    if local_scene is not None:
        # Local scene USD, the scene-development workflow: a freshly converted mesh or splat, not yet
        # registered. What it *is* is the USD's own business: the launch glue routes by the physics
        # schemas the USD authors, or their absence. Used in place the same way as a local vehicle
        # USD, with no cache copy; the receipt stays honest, with a sha256 the same way as a registry
        # asset. ``start`` is registry data: a local scene flies from its own origin, and
        # scripts/assets/spawn_site.py helps pick one.
        import hashlib

        from .models import AssetRef
        from .registry import Scene

        scene_sha = hashlib.sha256(local_scene.read_bytes()).hexdigest()
        scene = Scene(usd=AssetRef(url=local_scene.as_uri(), sha256=scene_sha, filename=local_scene.name))
    elif scene_id not in registry.scenes:
        raise RegistryError(f"unknown scene {scene_id!r}; have {list(registry.scenes)}")
    else:
        scene = registry.scenes[scene_id]

    veh_path = local if local is not None else (_resolve_asset(variant.usd, cache_dir) if fetch else None)
    if local_scene is not None:
        scn_path = local_scene  # in place: sibling files such as a mesh's textures/ must stay resolvable
    else:
        scn_path = _resolve_asset(scene.usd, cache_dir) if (fetch and scene.usd) else None
    # The geodetic origin: a launch override wins, else the registry scene's default, which can be None.
    geodetic_origin = launch.geodetic_origin or scene.geodetic_origin
    tested = TestedConfig(
        vehicle=variant.name,
        registry=str(source) if source is not None else None,
        vehicle_usd=variant.usd,
        px4=variant.px4,
        scene=scene_id,
        scene_usd=scene.usd,
        scene_start=scene.start,
        geodetic_origin=geodetic_origin,
        control=launch.control,
        runtime=launch.runtime,
        sensors=launch.sensors,
        environment=launch.environment,
    )
    return ResolvedLaunch(
        tested_config=tested,
        vehicle_usd_path=str(veh_path) if veh_path is not None else None,
        scene_usd_path=str(scn_path) if scn_path is not None else None,
    )
