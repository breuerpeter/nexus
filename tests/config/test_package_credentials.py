"""A package that adds a catalog supplies the credentials for its catalog's `s3://` assets through an
`s3_credentials` attribute of the module its `nexus.catalog` entry point names. The framework calls it at
the fetch of such an asset, never at load and never for another catalog's assets.
"""

import importlib
from pathlib import Path

import pytest

from nexus_uav import LaunchConfig
from nexus_uav._src.config import load_catalog, resolve

# The bytes the stand-in bucket serves, and their sha256.
DATA = b"astro_max usd bytes"
DATA_SHA = "f64912b37b15da27a2aa364a89fb43ceb77550d414b81b51ce7179f2e0cf97f6"
KEY = "bkt/catalog/assets/usd/vehicles/pkg_quad.usdz"
URL = f"s3://{KEY}"

# A package module whose credential attribute counts its calls and hands out the stand-in bucket's keys.
COUNTING = """
calls = []


def s3_credentials():
    calls.append(1)
    return {"aws_access_key_id": "test-key", "aws_secret_access_key": "test-secret"}
"""

# A package module whose credential attribute fails, as a refused token exchange would.
REFUSING = """
def s3_credentials():
    raise RuntimeError("token exchange refused")
"""


def _s3_vehicle(name: str) -> str:
    """A catalog naming one vehicle on the stand-in bucket."""
    return f'vehicles:\n  {name}:\n    usd: {{ url: "{URL}", sha256: "{DATA_SHA}" }}\n'


def _scene(tmp_path) -> str:
    """A local scene file, so a run fetches nothing but its vehicle."""
    scene = tmp_path / "ground.usda"
    scene.write_text("#usda 1.0\n")
    return str(scene)


def _in_empty_folder(tmp_path, monkeypatch) -> None:
    folder = tmp_path / "anywhere"
    folder.mkdir()
    monkeypatch.chdir(folder)


def _no_credentials_in_the_environment(monkeypatch) -> None:
    monkeypatch.delenv("AWS_ACCESS_KEY_ID")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY")


def _fetch_vehicle(tmp_path, name: str) -> bytes | str:
    """The fetched vehicle file's bytes, or the error's message when the fetch fails."""
    try:
        path = resolve(LaunchConfig(vehicle=name, scene=_scene(tmp_path)), cache_dir=tmp_path / "cache")
    except RuntimeError as exc:
        return str(exc)
    return Path(path.vehicle_usd_path).read_bytes()


def test_a_packages_credential_attribute_supplies_its_s3_credentials_at_the_fetch_not_at_load(
    tmp_path, monkeypatch, install_package, bucket
):
    """A package's credential attribute supplies the credentials for its catalog's `s3://` asset, and the
    framework calls it at the fetch, not at catalog load.

    Given an installed package whose catalog names a vehicle on the stand-in bucket, no Amazon Web Services (AWS) credentials in
    the environment, and a credential attribute that returns the bucket's keys and counts its calls; when
    `load_catalog()` runs, the count is 0; when `resolve` fetches the vehicle, the asset lands in the cache
    and the count is 1.
    """
    bucket[KEY] = DATA
    install_package("acme_drones", {"__init__.py": COUNTING, "nexus.catalog.yaml": _s3_vehicle("pkg_quad")})
    _no_credentials_in_the_environment(monkeypatch)
    _in_empty_folder(tmp_path, monkeypatch)

    load_catalog()
    at_load = list(importlib.import_module("acme_drones").calls)
    fetched = _fetch_vehicle(tmp_path, "pkg_quad")
    at_fetch = list(importlib.import_module("acme_drones").calls)

    assert (at_load, fetched, at_fetch) == ([], DATA, [1])


def test_a_run_that_fetches_only_another_catalogs_assets_never_calls_a_packages_credential_attribute(
    tmp_path, monkeypatch, install_package, bucket
):
    """A run that fetches only another catalog's assets never calls a package's credential attribute.

    Given package A with a counting credential attribute, and package B with an `s3://` vehicle and
    credentials in the environment; when `resolve` fetches B's vehicle, A's count is 0.
    """
    bucket[KEY] = DATA
    install_package("acme_drones", {"__init__.py": COUNTING, "nexus.catalog.yaml": "vehicles: {}\n"})
    install_package("bravo_drones", {"nexus.catalog.yaml": _s3_vehicle("bravo_quad")})
    _in_empty_folder(tmp_path, monkeypatch)

    fetched = _fetch_vehicle(tmp_path, "bravo_quad")

    assert (fetched, importlib.import_module("acme_drones").calls) == (DATA, [])


def test_an_error_the_credential_attribute_raises_fails_the_fetch_naming_the_assets_url(
    tmp_path, monkeypatch, install_package, bucket
):
    """An error the credential attribute raises fails the fetch, and the error names the asset's URL and
    carries the attribute's error.

    Given a package whose credential attribute raises `RuntimeError("token exchange refused")`; when
    `resolve` fetches its `s3://` vehicle, the raised error names the asset's URL and chains the attribute's
    error.
    """
    bucket[KEY] = DATA
    install_package("acme_drones", {"__init__.py": REFUSING, "nexus.catalog.yaml": _s3_vehicle("pkg_quad")})
    _no_credentials_in_the_environment(monkeypatch)
    _in_empty_folder(tmp_path, monkeypatch)

    with pytest.raises(Exception) as caught:
        resolve(LaunchConfig(vehicle="pkg_quad", scene=_scene(tmp_path)), cache_dir=tmp_path / "cache")

    assert (URL in str(caught.value), str(caught.value.__cause__)) == (True, "token exchange refused")


def test_a_package_catalog_with_no_credential_attribute_fetches_through_the_default_chain(
    tmp_path, monkeypatch, install_package, bucket
):
    """A package catalog that names no credential attribute fetches its `s3://` asset through boto3's
    default chain, as today.

    Given an installed package with an `s3://` vehicle, no credential attribute, and the bucket's keys in
    the environment; when `resolve` fetches, the asset lands in the cache.
    """
    bucket[KEY] = DATA
    install_package("acme_drones", {"nexus.catalog.yaml": _s3_vehicle("pkg_quad")})
    _in_empty_folder(tmp_path, monkeypatch)

    assert _fetch_vehicle(tmp_path, "pkg_quad") == DATA
