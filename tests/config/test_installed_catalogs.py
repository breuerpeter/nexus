"""A package installed beside the framework adds a catalog layer: the `nexus.catalog.yaml` in the folder of
the package a `nexus.catalog` entry point names. The layers go: the bundled catalog, then the installed
packages, then the project's.
"""

import importlib.resources
import json
import logging
from pathlib import Path

import pytest

from nexus_uav import LaunchConfig
from nexus_uav._src.config import CatalogError, NoMatchError, load_catalog, resolve

BUNDLED_ASTRO_SHA = "51665e5e6542b7f2ea80b897f7296ca5f1d2b18b1530f4ac532bf237fac4d0cd"
PACKAGE_SHA = "1111111111111111111111111111111111111111111111111111111111111111"
PROJECT_SHA = "2222222222222222222222222222222222222222222222222222222222222222"


def _vehicle(name: str, url: str, sha256: str = "0") -> str:
    """A catalog naming one vehicle at a full URL."""
    return f'vehicles:\n  {name}:\n    usd: {{ url: "{url}", sha256: "{sha256}" }}\n'


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def _empty_folder(tmp_path, monkeypatch) -> Path:
    """A working folder with no project catalog in it or over it."""
    folder = tmp_path / "anywhere"
    folder.mkdir()
    monkeypatch.chdir(folder)
    return folder


def _project(tmp_path, monkeypatch, catalog: str) -> Path:
    """A project folder holding ``catalog`` as its `nexus.catalog.yaml`, as the working folder."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "nexus.catalog.yaml").write_text(catalog)
    monkeypatch.chdir(project)
    return project


def test_a_vehicle_an_installed_package_names_flies_with_no_catalog_path_from_any_folder(
    tmp_path, monkeypatch, install_package
):
    """A vehicle an installed package's catalog names flies with no catalog path, from a directory with no
    project catalog.

    Given a package installed with a `nexus.catalog` entry point whose `nexus.catalog.yaml` names vehicle
    `pkg_quad`, when a run in an empty directory names `pkg_quad`, then it resolves to the package's
    Universal Scene Description (USD) file.
    """
    install_package("acme_drones", {"nexus.catalog.yaml": _vehicle("pkg_quad", "file:///acme/pkg_quad.usda")})
    _empty_folder(tmp_path, monkeypatch)

    try:
        url = resolve(LaunchConfig(vehicle="pkg_quad", scene="empty"), fetch=False).tested_config.vehicle_usd.url
    except NoMatchError as exc:
        url = str(exc)

    assert url == "file:///acme/pkg_quad.usda"


def test_the_bundled_vehicles_and_scenes_still_resolve_beside_an_installed_package(
    tmp_path, monkeypatch, install_package
):
    """A bundled vehicle and scene still resolve with a package installed.

    Given that package installed, when a run names a bundled vehicle and scene, then both resolve to the
    bundled entries, beside the package's own vehicle.
    """
    install_package("acme_drones", {"nexus.catalog.yaml": _vehicle("pkg_quad", "file:///acme/pkg_quad.usda")})
    _empty_folder(tmp_path, monkeypatch)

    catalog = load_catalog()

    assert (
        "pkg_quad" in catalog.vehicles,
        catalog.vehicles["astro_max_base"].usd.sha256,
        "empty" in catalog.scenes,
    ) == (True, BUNDLED_ASTRO_SHA, True)


def test_a_package_entry_with_a_bundled_name_replaces_it_and_warns_with_both_hashes(
    tmp_path, monkeypatch, caplog, install_package
):
    """A package entry that names a bundled entry replaces it, and the load warns with both hashes.

    Given a package catalog naming bundled vehicle `astro_max_base` with another sha256, when the catalog
    loads, then the package's entry resolves and the warning names both hashes.
    """
    install_package(
        "acme_drones", {"nexus.catalog.yaml": _vehicle("astro_max_base", "file:///acme/astro.usda", PACKAGE_SHA)}
    )
    _empty_folder(tmp_path, monkeypatch)

    with caplog.at_level(logging.WARNING):
        sha = load_catalog().vehicles["astro_max_base"].usd.sha256
    named = [all(s in m for s in ("astro_max_base", PACKAGE_SHA, BUNDLED_ASTRO_SHA)) for m in _warnings(caplog)]

    assert (sha, named) == (PACKAGE_SHA, [True])


def test_two_installed_packages_that_define_one_name_fail_the_load_naming_it_and_both(
    tmp_path, monkeypatch, install_package
):
    """Two installed packages that define one name fail the load, and the message names the name and both
    packages.

    Given two packages whose catalogs both name vehicle `pkg_quad`, when the catalog loads, then it fails
    and the message names `pkg_quad` and both packages.
    """
    install_package("acme_drones", {"nexus.catalog.yaml": _vehicle("pkg_quad", "file:///acme/pkg_quad.usda")})
    install_package("bravo_drones", {"nexus.catalog.yaml": _vehicle("pkg_quad", "file:///bravo/pkg_quad.usda")})
    _empty_folder(tmp_path, monkeypatch)

    with pytest.raises(CatalogError) as caught:
        load_catalog()

    assert [s in str(caught.value) for s in ("pkg_quad", "acme_drones", "bravo_drones")] == [True, True, True]


def test_a_project_entry_replaces_an_installed_packages_entry_and_warns_with_both_hashes(
    tmp_path, monkeypatch, caplog, install_package
):
    """A project entry replaces an installed package's entry of the same name, and the load warns with both
    hashes.

    Given a package naming `pkg_quad` and a project catalog naming `pkg_quad` with another sha256, when a
    run in the project names it, then the project's entry resolves and the warning names both hashes.
    """
    install_package(
        "acme_drones", {"nexus.catalog.yaml": _vehicle("pkg_quad", "file:///acme/pkg_quad.usda", PACKAGE_SHA)}
    )
    _project(tmp_path, monkeypatch, _vehicle("pkg_quad", "file:///project/pkg_quad.usda", PROJECT_SHA))

    with caplog.at_level(logging.WARNING):
        sha = resolve(LaunchConfig(vehicle="pkg_quad", scene="empty"), fetch=False).tested_config.vehicle_usd.sha256
    named = [all(s in m for s in ("pkg_quad", PACKAGE_SHA, PROJECT_SHA)) for m in _warnings(caplog)]

    assert (sha, named) == (PROJECT_SHA, [True])


def test_a_packages_compact_entry_completes_against_the_packages_own_base(tmp_path, monkeypatch, install_package):
    """A package's compact entries complete against the package's own `assets.base`, not the project's.

    Given a package catalog with its own `assets.base` and a compact `{name, sha256}` entry, and a project
    catalog with another base, when the catalog loads, then the entry's URL starts with the package's base.
    """
    install_package(
        "acme_drones",
        {
            "nexus.catalog.yaml": (
                "assets:\n  base: https://acme.example/catalog\n"
                "vehicles:\n  pkg_quad:\n    usd: { name: pkg_quad, sha256: abc }\n"
            )
        },
    )
    _project(
        tmp_path,
        monkeypatch,
        "assets:\n  base: s3://example-bucket/catalog\nvehicles:\n  project_vehicle:\n    usd: { name: project_vehicle, sha256: def }\n",
    )

    entry = load_catalog().vehicles.get("pkg_quad")

    assert (entry.usd.url if entry else None) == "https://acme.example/catalog/assets/usd/vehicles/pkg_quad-abc.usdz"


def test_an_entry_point_whose_package_holds_no_catalog_fails_the_load_naming_both(
    tmp_path, monkeypatch, install_package
):
    """A `nexus.catalog` entry point whose package holds no `nexus.catalog.yaml` fails the load, and the
    message names the entry point and the package.

    Given an entry point naming a package folder with pin files and no catalog file, when the catalog
    loads, then it fails with that message.
    """
    install_package("acme_pins", {"nexus.kit.ref": "registry.example/kit:8\n"}, entry="drones")
    _empty_folder(tmp_path, monkeypatch)

    with pytest.raises(CatalogError) as caught:
        load_catalog()

    assert [s in str(caught.value) for s in ("drones", "acme_pins")] == [True, True]


def test_the_receipt_names_each_catalog_layer_the_run_loaded_in_order(tmp_path, monkeypatch, install_package):
    """The receipt names each catalog layer the run loaded: the bundled one, each installed package, and the
    project's.

    Given a package installed and a project catalog, when a run builds, then the receipt names all three
    layers in load order, each with its package or path.

    The package installs as `acme-drones` from a folder named `acme_core`, so only the receipt's name, not
    the path, says which distribution to install to fly its entries again.
    """
    package = install_package(
        "acme_core",
        {"nexus.catalog.yaml": _vehicle("pkg_quad", "file:///acme/pkg_quad.usda")},
        distribution="acme-drones",
    )
    project = _project(tmp_path, monkeypatch, _vehicle("project_vehicle", "file:///project/vehicle.usda"))
    bundled = Path(str(importlib.resources.files("nexus_uav._src.config").joinpath("catalog.yaml")))

    receipt = json.loads(
        resolve(LaunchConfig(vehicle="project_vehicle", scene="empty"), fetch=False).tested_config.to_json()
    )

    assert receipt.get("catalogs") == [
        {"kind": "bundled", "name": "bundled", "path": str(bundled)},
        {"kind": "package", "name": "acme-drones", "path": str(package / "nexus.catalog.yaml")},
        {"kind": "project", "name": "project", "path": str(project / "nexus.catalog.yaml")},
    ]
