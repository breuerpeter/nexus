"""The vehicle and scene catalog a run flies: the one the wheel ships, and a project's beside it."""

import logging

import pytest

import nexus_sim
from nexus_sim import LaunchConfig
from nexus_sim._src.config import Catalog, CatalogError, load_catalog, resolve

PROJECT = """\
vehicles:
  project_vehicle:
    usd: { url: "file:///project_vehicle.usda", sha256: "0" }
"""


def test_a_run_inside_a_project_flies_the_nearest_catalog_beside_the_bundled_one(tmp_path, monkeypatch):
    """A run started inside a project flies the vehicles of the nearest `nexus.catalog.yaml` in its
    working directory or a directory over it, beside the bundled ones.

    Given a project directory whose `nexus.catalog.yaml` lists a vehicle at a local Universal Scene
    Description (USD) path, and a subdirectory of it as the working directory, when a run resolves
    that vehicle and `astro_max_base`, then both resolve, the first from the project's file.
    """
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "nexus.catalog.yaml").write_text(PROJECT)
    monkeypatch.chdir(project / "scripts")

    own = resolve(LaunchConfig(vehicle="project_vehicle", scene="empty"), fetch=False)
    bundled = resolve(LaunchConfig(vehicle="astro_max_base", scene="empty"), fetch=False)

    assert (own.tested_config.vehicle_usd.url, bundled.tested_config.vehicle) == (
        "file:///project_vehicle.usda",
        "astro_max_base",
    )


def _veh(name):
    return {"usd": {"url": f"file:///{name}", "sha256": "0"}}


def _catalog_of(*names, **kw):
    return Catalog.from_dict({"vehicles": {n: _veh(n) for n in names}, "scenes": {"empty": {}}, **kw})


def test_a_vehicle_resolves_by_its_key():
    catalog = _catalog_of("base", "fpv_lr1")
    assert catalog.vehicles["fpv_lr1"].usd.url == "file:///fpv_lr1"


SIBLING = """\
vehicles:
  sibling_vehicle:
    usd: { url: "file:///sibling.usdz", sha256: "0" }
"""


def test_a_catalog_beside_the_run_extends_the_one_in_the_wheel(tmp_path, monkeypatch):
    """A catalog beside the run extends the one in the wheel: `nexus.catalog.yaml` in the working
    directory or a parent adds its entries to the catalog a run flies, and where no parent holds one
    the run flies only the catalog the wheel ships.
    """
    project = tmp_path / "sibling"
    (project / "scripts").mkdir(parents=True)
    (project / "nexus.catalog.yaml").write_text(SIBLING)

    monkeypatch.chdir(project / "scripts")  # started below the file, which the walk up still finds
    beside = list(load_catalog().vehicles)
    monkeypatch.chdir(tmp_path)  # no parent of this one holds a catalog
    shipped = list(load_catalog().vehicles)

    assert ("sibling_vehicle" in beside, "sibling_vehicle" in shipped) == (True, False)


def test_an_entry_addresses_its_own_blob():
    """An entry addresses its own blob: a full URL fetches from where it says, and `{name, sha256}`
    resolves against its own catalog's base.
    """
    catalog = Catalog.from_dict(
        {
            "assets": {"base": "https://assets.example/catalog"},
            "vehicles": {
                "derived": {"usd": {"name": "astro", "sha256": "abc"}},
                "direct": {"usd": {"url": "s3://example-bucket/elsewhere/x.usdz", "sha256": "def"}},
            },
            "scenes": {"empty": {}},
        }
    )
    assert (catalog.vehicles["derived"].usd.url, catalog.vehicles["direct"].usd.url) == (
        "https://assets.example/catalog/assets/usd/vehicles/astro-abc.usdz",
        "s3://example-bucket/elsewhere/x.usdz",
    )


def test_the_shipped_catalog_resolves_exactly_as_today():
    """The shipped catalog resolves exactly as today: every vehicle and scene in the framework's own
    catalog keeps the URL it has now.
    """
    catalog = load_catalog()
    base = "https://d2837jz4fvtxko.cloudfront.net/public/assets/usd"
    hosted = [v.usd.url for v in catalog.vehicles.values()] + [s.usd.url for s in catalog.scenes.values() if s.usd]
    astro = catalog.vehicles["astro_max_base"].usd
    assert all(u.startswith(base) for u in hosted)
    assert astro.url == f"{base}/vehicles/astro_max_base-{astro.sha256}.usdz"


def test_the_shipped_catalog_names_two_vehicles():
    """The shipped catalog names two vehicles, `astro_max_base` and `astro_max_fpv`: the payload
    variants leave the tree, and a project that wants one brings it in its own catalog.
    """
    assert list(nexus_sim.Catalog.from_yaml().vehicles) == ["astro_max_base", "astro_max_fpv"]


def test_the_shipped_catalog_lists_a_hosted_scene_with_static_geometry():
    """The shipped catalog lists a photoreal scene with static geometry, hosted under the content delivery network's
    `public/` prefix: `powerline`, served from the scenes folder under its own name.
    """
    url = nexus_sim.Catalog.from_yaml().scenes["powerline"].usd.url
    assert url.startswith("https://d2837jz4fvtxko.cloudfront.net/public/assets/usd/scenes/powerline-")


HOSTED = "https://d2837jz4fvtxko.cloudfront.net/public/assets/usd/vehicles"


def test_the_two_astro_max_vehicles_keep_the_usds_they_fly_today():
    """`astro_max_base` and `astro_max_fpv` keep the USDs they fly today: the published USDs, pinned
    by hash, stay the source of record once the scripts that authored them leave the tree.
    """
    catalog = nexus_sim.Catalog.from_yaml()
    assert [catalog.vehicles[name].usd.url for name in ("astro_max_base", "astro_max_fpv")] == [
        f"{HOSTED}/astro_max_base-0055c7851d3efe08306f7dcdb9d19e2e634701ca01a5221ded1dabffc9cadb8a.usdz",
        f"{HOSTED}/astro_max_fpv-3a6953cbba38210d904126d0e769ae301a8f05c4891d88359ba3b42ea5bd1e89.usdz",
    ]


BUNDLED_ASTRO = f"{HOSTED}/astro_max_base-0055c7851d3efe08306f7dcdb9d19e2e634701ca01a5221ded1dabffc9cadb8a.usdz"
BUNDLED_EMPTY_SHA = "ab15e88be59c0ee93e63160c34b08485e3136d1f88313f24dcd67e482ecbed05"
REPIN_SHA = "1111111111111111111111111111111111111111111111111111111111111111"

HOSTED_PROJECT = """\
assets:
  base: s3://example-bucket/catalog
vehicles:
  project_vehicle:
    usd: { name: project_vehicle, sha256: abc }
"""

REPIN = f"""\
scenes:
  empty:
    usd: {{ url: "file:///empty.usdz", sha256: "{REPIN_SHA}" }}
"""


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_project_catalog_that_lists_only_its_own_entries_also_flies_the_bundled_ones(tmp_path, monkeypatch, caplog):
    """A project catalog that lists only its own entries also flies the bundled vehicles and scenes:
    a `nexus.catalog.yaml` with one vehicle and no scene resolves its vehicle, `astro_max_base` and
    `empty`, and no warning prints.
    """
    (tmp_path / "nexus.catalog.yaml").write_text(HOSTED_PROJECT)
    monkeypatch.chdir(tmp_path)

    with caplog.at_level(logging.WARNING):
        catalog = load_catalog()
        resolved = (
            "project_vehicle" in catalog.vehicles,
            "astro_max_base" in catalog.vehicles,
            "empty" in catalog.scenes,
        )

    assert (resolved, _warnings(caplog)) == ((True, True, True), [])


def test_each_entry_resolves_against_its_own_catalogs_base(tmp_path, monkeypatch):
    """Each entry resolves against its own catalog's base: a project entry under the project's
    `s3://` base, a bundled entry under the public CloudFront prefix.
    """
    (tmp_path / "nexus.catalog.yaml").write_text(HOSTED_PROJECT)
    monkeypatch.chdir(tmp_path)

    catalog = load_catalog()

    assert (catalog.vehicles["project_vehicle"].usd.url, catalog.vehicles["astro_max_base"].usd.url) == (
        "s3://example-bucket/catalog/assets/usd/vehicles/project_vehicle-abc.usdz",
        BUNDLED_ASTRO,
    )


def test_a_name_in_both_catalogs_takes_the_projects_entry_with_a_warning(tmp_path, monkeypatch, caplog):
    """A name in both catalogs: the project's entry wins, with a warning. A project catalog that
    redefines `empty` with another hash resolves `empty` to the project's hash, and one warning names
    `empty` and both hashes.
    """
    (tmp_path / "nexus.catalog.yaml").write_text(REPIN)
    monkeypatch.chdir(tmp_path)

    with caplog.at_level(logging.WARNING):
        sha = load_catalog().scenes["empty"].usd.sha256
    named = [all(s in m for s in ("empty", REPIN_SHA, BUNDLED_EMPTY_SHA)) for m in _warnings(caplog)]

    assert (sha, named) == (REPIN_SHA, [True])


def test_a_catalog_named_by_path_also_extends_the_bundled_one(tmp_path, monkeypatch):
    """A catalog named by path also extends the bundled one: a file with one vehicle, loaded by
    `load_catalog(path)`, resolves the bundled `astro_max_base` too.
    """
    named = tmp_path / "project.yaml"
    named.write_text(HOSTED_PROJECT)
    monkeypatch.chdir(tmp_path)

    assert load_catalog(named).vehicles["astro_max_base"].usd.url == BUNDLED_ASTRO


def test_a_catalog_entry_that_still_carries_px4_fails_to_load(tmp_path):
    """A catalog entry that still carries `px4:` fails to load: the airframe lives in the vehicle's Universal Scene Description (USD) file.

    Given a project catalog whose vehicle entry has `px4: { airframe: astro_max }`, when
    `nexus_sim.Catalog.from_yaml(path)` loads it, then it raises and names the `px4` field.
    """
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        "vehicles:\n"
        "  project_vehicle:\n"
        '    usd: { url: "file:///project_vehicle.usda", sha256: abc }\n'
        "    px4: { airframe: astro_max }\n"
        "scenes:\n  empty: {}\n"
    )
    with pytest.raises(ValueError, match=r"vehicles\.project_vehicle\.px4"):
        nexus_sim.Catalog.from_yaml(catalog)


MY_QUAD = """\
vehicles:
  my_quad:
    usd: { url: "file:///my_quad.usda", sha256: abc }
"""


def test_a_catalog_with_a_defaults_block_fails_to_load_and_says_to_name_the_vehicle_and_scene(tmp_path):
    """A project catalog with a `defaults` block fails to load, and the error names the removal and
    says to name the vehicle and scene on the run.

    Given a catalog with `defaults: { vehicle: my_quad }`, when `nexus_sim.Catalog.from_yaml(path)` reads
    it, then it raises an error that names `defaults` as removed and points at `--vehicle` and `--scene`.
    """
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(MY_QUAD + "defaults: { vehicle: my_quad }\n")

    with pytest.raises((CatalogError, ValueError)) as e:
        nexus_sim.Catalog.from_yaml(catalog)

    assert [w in str(e.value) for w in ("defaults", "remov", "--vehicle", "--scene")] == [True] * 4


def test_a_project_catalog_that_lists_only_what_it_adds_loads_on_its_own(tmp_path):
    """A project catalog that lists only what it adds loads on its own.

    Given a file with one vehicle, `scenes: {}` and no `defaults`, when `nexus_sim.Catalog.from_yaml(path)`
    reads it, then it returns a catalog with that one vehicle and no error.
    """
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(MY_QUAD + "scenes: {}\n")

    assert list(nexus_sim.Catalog.from_yaml(catalog).vehicles) == ["my_quad"]


BUNDLED_ASTRO_SHA = "0055c7851d3efe08306f7dcdb9d19e2e634701ca01a5221ded1dabffc9cadb8a"
BUNDLED_FPV_SHA = "3a6953cbba38210d904126d0e769ae301a8f05c4891d88359ba3b42ea5bd1e89"

KEYED_REPIN = f"""\
vehicles:
  astro_max_base:
    usd: {{ url: "file:///astro_max_base.usdz", sha256: "{REPIN_SHA}" }}
"""

KEYED_PROJECT = """\
vehicles:
  project_vehicle:
    usd: { url: "file:///project_vehicle.usdz", sha256: abc }
"""


def test_a_project_vehicle_keyed_by_a_bundled_name_replaces_the_bundled_entry_with_a_warning(
    tmp_path, monkeypatch, caplog
):
    """A project vehicle keyed by a bundled name replaces the bundled entry and warns with both hashes.

    Given a project catalog with `vehicles: { astro_max_base: { usd: … } }`, when `load_catalog`
    loads it, then `astro_max_base` resolves to the project's USD, `astro_max_fpv` still resolves to
    the bundled one, and one warning names both sha256 values.
    """
    (tmp_path / "nexus.catalog.yaml").write_text(KEYED_REPIN)
    monkeypatch.chdir(tmp_path)

    with caplog.at_level(logging.WARNING):
        catalog = load_catalog()
    shas = (catalog.vehicles["astro_max_base"].usd.sha256, catalog.vehicles["astro_max_fpv"].usd.sha256)
    named = [all(s in m for s in ("astro_max_base", REPIN_SHA, BUNDLED_ASTRO_SHA)) for m in _warnings(caplog)]

    assert (shas, named) == ((REPIN_SHA, BUNDLED_FPV_SHA), [True])


def test_a_project_catalog_that_adds_one_vehicle_by_key_still_flies_the_bundled_ones(tmp_path, monkeypatch):
    """A project catalog that adds one vehicle by key still flies the bundled vehicles and scenes.

    Given a project catalog with only `vehicles: { project_vehicle: … }`, when `load_catalog` loads
    it, then `project_vehicle`, `astro_max_base`, `astro_max_fpv` and the scene `empty` all resolve.
    """
    (tmp_path / "nexus.catalog.yaml").write_text(KEYED_PROJECT)
    monkeypatch.chdir(tmp_path)

    catalog = load_catalog()
    resolved = [name in catalog.vehicles for name in ("project_vehicle", "astro_max_base", "astro_max_fpv")]

    assert (resolved, "empty" in catalog.scenes) == ([True, True, True], True)
