"""The checked-in catalog + name resolution.

Each vehicle variant carries a ``name`` handle, its Universal Scene Description (USD) and its PX4
spec; a launch names the variant it flies, and a launch that names none takes ``defaults.vehicle``.
No per-key override/transform machinery: that "clever" path is on hold.
"""

from __future__ import annotations

import pathlib

import yaml
from pydantic import Field, model_validator

from nexus._src.assets.resolver import hosted_url
from nexus._src.core import logger

from .models import AssetRef, GeodeticOrigin, Px4Spec, _Base


def _expand_usd(data, kind: str, base: str | None):
    """Expand a registry ``usd: {name, sha256}`` into a full :class:`AssetRef`, with the
    content-addressed ``.usdz`` URL + cache filename derived from the registry's ``assets.base``.

    An explicit ``{url, ...}`` ref passes through untouched, whether it names an asset hosted
    elsewhere, a local ``file://`` one, or a test fixture. So a catalog names blobs in more than one
    place while the registry surface stays compact, and the resolver, receipt and cache keep working
    on ordinary ``AssetRef``s. The load reports a compact ref with no base to complete it.
    """
    if isinstance(data, dict) and isinstance(data.get("usd"), dict):
        usd = data["usd"]
        if "name" in usd and "url" not in usd:
            name, sha = usd["name"], usd.get("sha256", "")
            if not base:
                raise RegistryError(f"{kind[:-1]} {name!r} names no url and the registry has no assets.base")
            data = {
                **data,
                "usd": {"url": hosted_url(base, kind, name, sha), "sha256": sha, "filename": f"{name}.usdz"},
            }
    return data


class RegistryError(Exception):
    """A malformed registry: a duplicate name, a dangling reference, and so on."""


class NoMatchError(RegistryError):
    """No vehicle variant carries the name the launch asks for."""


class VehicleVariant(_Base):
    """One catalog entry: a `name` handle + its Universal Scene Description (USD) + PX4."""

    name: str
    usd: AssetRef
    px4: Px4Spec | None = None


class Scene(_Base):
    """Static-geometry asset + geo origin. ``usd: null`` = flat ground.

    The scene USD is the single authority on its role: a USD that authors ``UsdPhysics``
    collision schemas is simulation geometry for the Newton ``ModelBuilder``, for example the slalom
    pillars; a USD without physics schemas is the visual world, referenced by the renderer,
    never the builder, since a colliding photogrammetry/splat USD is the known model-build failure.
    The geo origin anchors the environment either way.
    """

    usd: AssetRef | None = None
    geodetic_origin: GeodeticOrigin | None = None
    start: tuple[float, float, float] | None = None
    """The scene-frame point placed at the world origin: where the vehicle starts, on the z=0
    physics ground. ``None`` = the scene's own origin. Placement is per-scene registry data, NOT
    baked into the asset; ``scripts/assets/spawn_site.py`` suggests a value for a converted scan.
    """


class Defaults(_Base):
    """What a launch that names neither a vehicle nor a scene flies."""

    vehicle: str | None = None
    scene: str = "empty"


class Assets(_Base):
    """Where a catalog's blobs live: the base a compact ``{name, sha256}`` entry completes against."""

    base: str | None = None
    """The base URL, for example a Content Delivery Network (CDN) prefix or an ``s3://`` one.

    ``None`` means every entry carries a full URL of its own.
    """


class Registry(_Base):
    """The checked-in vehicle/scene catalog, where its blobs live, and the defaults a bare launch flies.

    Resolution is by name: a launch names a :class:`VehicleVariant` and one that names none takes
    ``defaults.vehicle``. Built from the bundled ``registry.yaml`` via :func:`load_registry` in the
    common case, or from a catalog a project beside the framework keeps.
    """

    assets: Assets = Field(default_factory=Assets)
    """Where this catalog's blobs live; see :class:`Assets`."""
    vehicles: list[VehicleVariant] = Field(default_factory=list)
    """The catalog of vehicle variants, each under its ``name`` handle.

    Empty by default; an empty catalog resolves no name.
    """
    scenes: dict[str, Scene] = Field(default_factory=dict)
    """The available scenes, keyed by name (the key a ``LaunchConfig.scene`` refers to).

    Empty by default.
    """
    defaults: Defaults = Field(default_factory=Defaults)
    """What a launch that names neither a vehicle nor a scene flies.

    Defaults to a :class:`Defaults` with no vehicle and the ``"empty"`` scene.
    """

    @model_validator(mode="before")
    @classmethod
    def _expand_refs(cls, data):
        """Complete every compact ``usd`` ref against ``assets.base`` before the entries validate.

        It happens here rather than on each entry because the base is the catalog's, not the entry's.
        """
        if not isinstance(data, dict):
            return data
        base = (data.get("assets") or {}).get("base") if isinstance(data.get("assets"), dict) else None
        out = dict(data)
        if isinstance(out.get("vehicles"), list):
            out["vehicles"] = [_expand_usd(v, "vehicles", base) for v in out["vehicles"]]
        if isinstance(out.get("scenes"), dict):
            out["scenes"] = {k: _expand_usd(s, "scenes", base) for k, s in out["scenes"].items()}
        return out

    @classmethod
    def from_dict(cls, data: dict | None) -> Registry:
        """Build and validate a registry from a plain dict.

        Args:
            data: The mapping of registry fields. ``None``, or an empty dict, yields an empty,
                fully defaulted registry.

        Returns:
            The validated ``Registry``.

        Raises:
            pydantic.ValidationError: If ``data`` has unknown keys or values that fail
                validation.
            RegistryError: If the catalog fails load-time validation; see :meth:`validate`.
        """
        reg = cls.model_validate(data or {})
        reg.validate()
        return reg

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path | None = None) -> Registry:
        """Build and validate a registry from a YAML file.

        Args:
            path: Path to the registry YAML. ``None`` loads the package's bundled
                ``registry.yaml``.

        Returns:
            The validated ``Registry``.

        Raises:
            pydantic.ValidationError: If the parsed document fails validation.
            RegistryError: If the catalog fails load-time validation; see :meth:`validate`.
        """
        path = pathlib.Path(path) if path is not None else _DEFAULT_REGISTRY
        return cls.from_dict(yaml.safe_load(path.read_text()))

    def validate(self) -> None:
        """Load-time validation, so a checked-in registry can't ship broken:

        * two variants with the same `name`: resolution would be non-deterministic; reject.
        * the **default scene must exist**, and a **default vehicle** must name one of the
          catalog's variants, else a dangling reference.
        """
        names = [v.name for v in self.vehicles]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise RegistryError(f"duplicate vehicle name(s): {dupes}")
        if self.defaults.scene not in self.scenes:
            raise RegistryError(f"defaults.scene {self.defaults.scene!r} is not a defined scene {list(self.scenes)}")
        if self.defaults.vehicle is not None and self.defaults.vehicle not in names:
            raise RegistryError(f"defaults.vehicle {self.defaults.vehicle!r} is not a defined vehicle {names}")

    def by_name(self, name: str) -> VehicleVariant:
        """Look up a variant by its `name` handle: the `--vehicle <name>` path."""
        for v in self.vehicles:
            if v.name == name:
                return v
        raise NoMatchError(f"no vehicle named {name!r}; registry names: {[v.name for v in self.vehicles]}")


_DEFAULT_REGISTRY = pathlib.Path(__file__).parent / "registry.yaml"
REGISTRY_FILENAME = "nexus.registry.yaml"


def discover_registry(start: str | pathlib.Path | None = None) -> pathlib.Path | None:
    """The nearest ``nexus.registry.yaml`` in *start* or a directory over it, the working directory
    by default.

    A project beside the framework keeps its catalog in its own tree, so a run started anywhere
    inside it flies that catalog with no flag and no variable, the same walk up that `uv`, `ruff`
    and `pytest` do for their own files.

    Returns:
        The first file the walk up finds, or ``None`` when no directory holds one.
    """
    here = pathlib.Path(start or pathlib.Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / REGISTRY_FILENAME
        if candidate.is_file():
            return candidate
    return None


def registry_path(path: str | pathlib.Path | None = None) -> pathlib.Path:
    """The catalog a run flies on top of the bundled one: *path* when it names one, else the nearest
    discovered one, else the catalog bundled in the wheel. Naming the file explicitly always wins over
    the walk up.
    """
    if path is not None:
        return pathlib.Path(path)
    return discover_registry() or _DEFAULT_REGISTRY


def _sha(usd: AssetRef | None) -> str | None:
    return usd.sha256 if usd is not None else None


def _extend(bundled: Registry, project: Registry) -> Registry:
    """*project* laid over *bundled*: a project entry replaces the bundled entry of the same name, and
    each replacement warns with both hashes, so a stale copy of a bundled entry never flies silently.

    Both catalogs have completed their compact refs against their own ``assets.base`` already, so
    each entry keeps its own catalog's URL. The project's ``defaults`` win key by key, and its
    ``assets`` stays the catalog's, since that base is where the project's new blobs go.
    """
    shipped = {v.name: v for v in bundled.vehicles}
    for v in project.vehicles:
        if v.name in shipped:
            logger.warning(
                f"vehicle {v.name!r}: the project's entry (sha256 {_sha(v.usd)}) replaces the bundled one "
                f"(sha256 {_sha(shipped[v.name].usd)})"
            )
    for name, scene in project.scenes.items():
        if name in bundled.scenes:
            logger.warning(
                f"scene {name!r}: the project's entry (sha256 {_sha(scene.usd)}) replaces the bundled one "
                f"(sha256 {_sha(bundled.scenes[name].usd)})"
            )
    own = {v.name for v in project.vehicles}
    return Registry(
        assets=project.assets,
        vehicles=[v for v in bundled.vehicles if v.name not in own] + project.vehicles,
        scenes={**bundled.scenes, **project.scenes},
        defaults=bundled.defaults.model_copy(
            update=project.defaults.model_dump(include=project.defaults.model_fields_set)
        ),
    )


def load_registry(path: str | pathlib.Path | None = None) -> Registry:
    """Load + validate the bundled catalog, extended by the one :func:`registry_path` picks for *path*.

    A project catalog lists only what it adds; see :func:`_extend` for how a name in both resolves.
    Validation runs on the merged catalog, so a project entry may name a bundled scene as its default.
    """
    bundled = Registry.from_yaml(_DEFAULT_REGISTRY)
    source = registry_path(path)
    if source.resolve() == _DEFAULT_REGISTRY.resolve():
        return bundled
    reg = _extend(bundled, Registry.model_validate(yaml.safe_load(source.read_text()) or {}))
    reg.validate()
    return reg
