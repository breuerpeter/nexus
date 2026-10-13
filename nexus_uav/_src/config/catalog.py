"""The checked-in catalog + name resolution.

The catalog keys each vehicle variant and each scene by name; a launch names the variant it flies
and the scene it flies in.
No per-key override/transform machinery: that "clever" path is on hold.
"""

from __future__ import annotations

import importlib.resources
import pathlib
from importlib.metadata import entry_points
from typing import Literal

import yaml
from pydantic import Field, model_validator

from nexus_uav._src.assets.resolver import hosted_url
from nexus_uav._src.assets.resolver import resolve as fetch_asset
from nexus_uav._src.core import logger

from .models import AssetRef, _Base


def _expand_usd(data, kind: str, base: str | None):
    """Expand a catalog ``usd: {name, sha256}`` into a full :class:`AssetRef`, with the
    content-addressed ``.usdz`` URL + cache filename derived from the catalog's ``assets.base``.

    An explicit ``{url, ...}`` ref passes through untouched, whether it names an asset hosted
    elsewhere, a local ``file://`` one, or a test fixture. So a catalog names blobs in more than one
    place while the catalog surface stays compact, and the resolver, receipt and cache keep working
    on ordinary ``AssetRef``s. The load reports a compact ref with no base to complete it.
    """
    if isinstance(data, dict) and isinstance(data.get("usd"), dict):
        usd = data["usd"]
        if "name" in usd and "url" not in usd:
            name, sha = usd["name"], usd.get("sha256", "")
            if not base:
                raise CatalogError(f"{kind[:-1]} {name!r} names no url and the catalog has no assets.base")
            data = {
                **data,
                "usd": {"url": hosted_url(base, f"usd/{kind}", name, sha), "sha256": sha, "filename": f"{name}.usdz"},
            }
    return data


class CatalogError(Exception):
    """A malformed catalog: a dangling reference, an unknown scene, and so on."""


class NoMatchError(CatalogError):
    """No vehicle variant carries the name the launch asks for."""


class VehicleVariant(_Base):
    """One catalog entry: a Universal Scene Description (USD) file, which declares its controller.

    The entry's name is its key in :attr:`Catalog.vehicles`.
    """

    usd: AssetRef


class Scene(_Base):
    """A scene: its Universal Scene Description (USD) file, and where the vehicle starts in it.

    The scene USD is the single authority on its role: a USD that authors ``UsdPhysics``
    collision schemas is simulation geometry for the Newton ``ModelBuilder``, for example the slalom
    pillars; a USD without physics schemas is the visual world, referenced by the renderer,
    never the builder, since a colliding photogrammetry/splat USD is the known model-build failure.
    Its ``PhysicsScene`` prim authors where it sits on Earth either way, see
    ``nexus_uav._src.scene.world``.
    """

    usd: AssetRef
    start: tuple[float, float, float] | None = None
    """The scene-frame point placed at the world origin: where the vehicle starts, on the z=0
    physics ground. ``None`` = the scene's own origin. Placement is per-scene catalog data, NOT
    baked into the asset; ``scripts/assets/spawn_site.py`` suggests a value for a converted scan.
    """


class Assets(_Base):
    """Where a catalog's blobs live: the base a compact ``{name, sha256}`` entry completes against."""

    base: str | None = None
    """The base URL, for example a Content Delivery Network (CDN) prefix or an ``s3://`` one.

    ``None`` means every entry carries a full URL of its own.
    """


class Catalog(_Base):
    """The checked-in vehicle/scene catalog and where its blobs live.

    Resolution is by name: a launch names a :class:`VehicleVariant` and a scene. Built by
    :func:`load_catalog` from the bundled ``catalog.yaml``, extended by the catalog of each
    installed package and by the catalog a project beside the framework keeps.
    """

    assets: Assets = Field(default_factory=Assets)
    """Where this catalog's blobs live; see :class:`Assets`."""
    vehicles: dict[str, VehicleVariant] = Field(default_factory=dict)
    """The catalog of vehicle variants, keyed by name (the key a ``LaunchConfig.vehicle`` refers to).

    Empty by default; an empty catalog resolves no name.
    """
    scenes: dict[str, Scene] = Field(default_factory=dict)
    """The available scenes, keyed by name (the key a ``LaunchConfig.scene`` refers to).

    Empty by default.
    """

    @model_validator(mode="before")
    @classmethod
    def _expand_refs(cls, data):
        """Complete every compact ``usd`` ref against ``assets.base`` before the entries validate.

        It happens here rather than on each entry because the base is the catalog's, not the entry's.
        """
        if not isinstance(data, dict):
            return data
        if "defaults" in data:
            raise ValueError(
                "the catalog's `defaults` block was removed: a run names its vehicle and scene, "
                "with `--vehicle` and `--scene` or `Sim(vehicle, scene=...)`"
            )
        base = (data.get("assets") or {}).get("base") if isinstance(data.get("assets"), dict) else None
        out = dict(data)
        for kind in ("vehicles", "scenes"):
            if isinstance(out.get(kind), dict):
                out[kind] = {k: _expand_usd(e, kind, base) for k, e in out[kind].items()}
        return out

    @classmethod
    def from_dict(cls, data: dict | None) -> Catalog:
        """Build and validate a catalog from a plain dict.

        Args:
            data: The mapping of catalog fields. ``None``, or an empty dict, yields an empty,
                fully defaulted catalog.

        Returns:
            The validated ``Catalog``.

        Raises:
            pydantic.ValidationError: If ``data`` has unknown keys or values that fail
                validation.
            CatalogError: If a compact ``usd`` entry has no ``assets.base`` to complete it.
        """
        return cls.model_validate(data or {})

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path | None = None) -> Catalog:
        """Build and validate a catalog from a YAML file.

        Args:
            path: Path to the catalog YAML. ``None`` loads the package's bundled
                ``catalog.yaml``.

        Returns:
            The validated ``Catalog``.

        Raises:
            pydantic.ValidationError: If the parsed document fails validation.
            CatalogError: If a compact ``usd`` entry has no ``assets.base`` to complete it.
        """
        path = pathlib.Path(path) if path is not None else _DEFAULT_CATALOG
        return cls.from_dict(yaml.safe_load(path.read_text()))

    def fetch_vehicle(self, name: str) -> pathlib.Path:
        """Fetch the named vehicle's Universal Scene Description (USD) file and return its local path.

        The file lands in the asset cache, which ``$NEXUS_ASSET_CACHE`` overrides, and a later call reads
        it from there with no download. A project that loads a catalog vehicle into a simulator of its
        own, such as Isaac Lab, opens the returned file.

        Args:
            name: The vehicle's key in :attr:`vehicles`, such as ``astro_max_base``.

        Returns:
            The path of the cached file, whose sha256 matches the catalog entry.

        Raises:
            NoMatchError: If the catalog names no vehicle ``name``. The message lists the vehicles it names.
            ValueError: If the fetched file's sha256 differs from the catalog entry's.
        """
        if name not in self.vehicles:
            raise NoMatchError(f"no vehicle named {name!r}; catalog names: {list(self.vehicles)}")
        return fetch_asset(self.vehicles[name].usd.model_dump(exclude_none=True))


_DEFAULT_CATALOG = pathlib.Path(__file__).parent / "catalog.yaml"
CATALOG_FILENAME = "nexus.catalog.yaml"
CATALOG_GROUP = "nexus.catalog"  # the entry-point group whose entries name the packages that add a catalog


def discover_catalog(start: str | pathlib.Path | None = None) -> pathlib.Path | None:
    """The nearest ``nexus.catalog.yaml`` in *start* or a directory over it, the working directory
    by default.

    A project beside the framework keeps its catalog in its own tree, so a run started anywhere
    inside it flies that catalog with no flag and no variable, the same walk up that `uv`, `ruff`
    and `pytest` do for their own files.

    Returns:
        The first file the walk up finds, or ``None`` when no directory holds one.
    """
    here = pathlib.Path(start or pathlib.Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / CATALOG_FILENAME
        if candidate.is_file():
            return candidate
    return None


class CatalogLayer(_Base):
    """One catalog a run loads, and its owner: the bundled one, an installed package's or the project's.

    Each layer but the bundled one keeps its pin files beside its catalog file.
    """

    kind: Literal["bundled", "package", "project"]
    name: str
    """The package's distribution name for a package layer, else the kind."""
    path: pathlib.Path
    """The layer's catalog file."""

    @property
    def label(self) -> str:
        """How a message names the layer: ``the bundled catalog``, ``package acme_drones``, ``the project``."""
        return {"bundled": "the bundled catalog", "project": "the project"}.get(self.kind, f"package {self.name}")


def _installed_layers() -> list[CatalogLayer]:
    """The catalog of each package a ``nexus.catalog`` entry point names, by entry point name.

    Raises:
        CatalogError: An entry point names a package whose folder holds no ``nexus.catalog.yaml``.
    """
    layers = []
    for entry in sorted(entry_points(group=CATALOG_GROUP), key=lambda e: e.name):
        package = entry.dist.name if entry.dist is not None else entry.module
        path = pathlib.Path(str(importlib.resources.files(entry.module))) / CATALOG_FILENAME
        if not path.is_file():
            raise CatalogError(
                f"the {CATALOG_GROUP} entry point {entry.name!r} of package {package} names {entry.module!r}, "
                f"whose folder holds no {CATALOG_FILENAME}"
            )
        layers.append(CatalogLayer(kind="package", name=package, path=path))
    return layers


def catalog_layers(path: str | pathlib.Path | None = None) -> list[CatalogLayer]:
    """The catalogs a run loads, in load order: the bundled one, then each installed package's, then the
    project's, which is *path* when it names one, else the nearest discovered one. Naming the file
    explicitly always wins over the walk up.

    A project file already loaded as a layer, such as the catalog of a package installed from the checkout
    the run starts in, loads once.

    Raises:
        CatalogError: An installed package holds no catalog file; see :func:`_installed_layers`.
    """
    layers = [CatalogLayer(kind="bundled", name="bundled", path=_DEFAULT_CATALOG), *_installed_layers()]
    project = pathlib.Path(path) if path is not None else discover_catalog()
    if project is not None and project.resolve() not in {layer.path.resolve() for layer in layers}:
        layers.append(CatalogLayer(kind="project", name="project", path=project))
    return layers


def _sha(usd: AssetRef | None) -> str | None:
    return usd.sha256 if usd is not None else None


def load_layers(layers: list[CatalogLayer]) -> Catalog:
    """Each of *layers* laid over the ones before it: an entry replaces the one of the same name below
    it, and each replacement warns with both hashes, so a stale copy never flies silently.

    Each layer completes its compact refs against its own ``assets.base``, so each entry keeps its own
    catalog's URL. The project's ``assets`` stays the catalog's, since that base is where the project's
    new blobs go; with no project, the bundled one's does.

    Raises:
        CatalogError: Two installed packages define one name, since no order between them means anything.
    """
    catalog = Catalog.from_yaml(layers[0].path)
    owners = {("vehicle", name): layers[0] for name in catalog.vehicles}
    owners |= {("scene", name): layers[0] for name in catalog.scenes}
    for layer in layers[1:]:
        own = Catalog.from_yaml(layer.path)
        for kind, below, entries in (
            ("vehicle", catalog.vehicles, own.vehicles),
            ("scene", catalog.scenes, own.scenes),
        ):
            for name, entry in entries.items():
                if (kind, name) in owners:
                    before = owners[(kind, name)]
                    if before.kind == layer.kind == "package":
                        raise CatalogError(
                            f"{kind} {name!r} is defined by two installed packages, {before.name} and {layer.name}"
                        )
                    logger.warning(
                        f"{kind} {name!r}: {layer.label}'s entry (sha256 {_sha(entry.usd)}) replaces "
                        f"{before.label}'s (sha256 {_sha(below[name].usd)})"
                    )
                owners[(kind, name)] = layer
        catalog = Catalog(
            assets=own.assets if layer.kind == "project" else catalog.assets,
            vehicles={**catalog.vehicles, **own.vehicles},
            scenes={**catalog.scenes, **own.scenes},
        )
    return catalog


def load_catalog(path: str | pathlib.Path | None = None) -> Catalog:
    """Load + validate the catalogs :func:`catalog_layers` lists for *path*, each over the ones before it.

    A package or project catalog lists only what it adds; see :func:`load_layers` for how a name in two
    resolves.
    """
    return load_layers(catalog_layers(path))
