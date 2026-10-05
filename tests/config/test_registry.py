"""Registry resolution: name lookup, load-time validation, and asset-reference expansion."""

import logging

import pytest

import nexus
from nexus._src.config import Registry, RegistryError, load_registry


def _veh(name):
    return {"usd": {"url": f"file:///{name}", "sha256": "0"}}


def _reg(*names, **kw):
    return Registry.from_dict({"vehicles": {n: _veh(n) for n in names}, "scenes": {"empty": {}}, **kw})


def test_a_vehicle_resolves_by_its_key():
    reg = _reg("base", "fpv_lr1")
    assert reg.vehicles["fpv_lr1"].usd.url == "file:///fpv_lr1"


SIBLING = """\
vehicles:
  sibling_vehicle:
    usd: { url: "file:///sibling.usdz", sha256: "0" }
"""


def test_a_registry_beside_the_run_extends_the_one_in_the_wheel(tmp_path, monkeypatch):
    """A registry beside the run extends the one in the wheel: `nexus.registry.yaml` in the working
    directory or a parent adds its entries to the catalog a run flies, and where no parent holds one
    the run flies only the catalog the wheel ships.
    """
    project = tmp_path / "sibling"
    (project / "scripts").mkdir(parents=True)
    (project / "nexus.registry.yaml").write_text(SIBLING)

    monkeypatch.chdir(project / "scripts")  # started below the file, which the walk up still finds
    beside = list(load_registry().vehicles)
    monkeypatch.chdir(tmp_path)  # no parent of this one holds a registry
    shipped = list(load_registry().vehicles)

    assert ("sibling_vehicle" in beside, "sibling_vehicle" in shipped) == (True, False)


def test_an_entry_addresses_its_own_blob():
    """An entry addresses its own blob: a full URL fetches from where it says, and `{name, sha256}`
    resolves against its own registry's base.
    """
    reg = Registry.from_dict(
        {
            "assets": {"base": "https://assets.example/catalog"},
            "vehicles": {
                "derived": {"usd": {"name": "astro", "sha256": "abc"}},
                "direct": {"usd": {"url": "s3://example-bucket/elsewhere/x.usdz", "sha256": "def"}},
            },
            "scenes": {"empty": {}},
        }
    )
    assert (reg.vehicles["derived"].usd.url, reg.vehicles["direct"].usd.url) == (
        "https://assets.example/catalog/assets/usd/vehicles/astro-abc.usdz",
        "s3://example-bucket/elsewhere/x.usdz",
    )


def test_the_shipped_catalog_resolves_exactly_as_today():
    """The shipped catalog resolves exactly as today: every vehicle and scene in the framework's own
    registry keeps the URL it has now.
    """
    reg = load_registry()
    base = "https://d2837jz4fvtxko.cloudfront.net/public/assets/usd"
    hosted = [v.usd.url for v in reg.vehicles.values()] + [s.usd.url for s in reg.scenes.values() if s.usd]
    astro = reg.vehicles["astro_max_base"].usd
    assert all(u.startswith(base) for u in hosted)
    assert astro.url == f"{base}/vehicles/astro_max_base-{astro.sha256}.usdz"


def test_the_shipped_catalog_names_two_vehicles():
    """The shipped catalog names two vehicles, `astro_max_base` and `astro_max_fpv`: the payload
    variants leave the tree, and a project that wants one brings it in its own catalog.
    """
    assert list(nexus.Registry.from_yaml().vehicles) == ["astro_max_base", "astro_max_fpv"]


def test_the_shipped_catalog_lists_a_hosted_scene_with_static_geometry():
    """The shipped catalog lists a photoreal scene with static geometry, hosted under the content delivery network's
    `public/` prefix: `powerline`, served from the scenes folder under its own name.
    """
    url = nexus.Registry.from_yaml().scenes["powerline"].usd.url
    assert url.startswith("https://d2837jz4fvtxko.cloudfront.net/public/assets/usd/scenes/powerline-")


HOSTED = "https://d2837jz4fvtxko.cloudfront.net/public/assets/usd/vehicles"


def test_the_two_astro_max_vehicles_keep_the_usds_they_fly_today():
    """`astro_max_base` and `astro_max_fpv` keep the USDs they fly today: the published USDs, pinned
    by hash, stay the source of record once the scripts that authored them leave the tree.
    """
    reg = nexus.Registry.from_yaml()
    assert [reg.vehicles[name].usd.url for name in ("astro_max_base", "astro_max_fpv")] == [
        f"{HOSTED}/astro_max_base-38700d1d05b739bba445b1b1d4fb35c7aec8ec4ef05d49d4dcb8b8941ef84ca9.usdz",
        f"{HOSTED}/astro_max_fpv-51cec8c50db9b50d9a8baa86730b780e574d1d8846c5483ffe92d168cf2cbbe8.usdz",
    ]


BUNDLED_ASTRO = f"{HOSTED}/astro_max_base-38700d1d05b739bba445b1b1d4fb35c7aec8ec4ef05d49d4dcb8b8941ef84ca9.usdz"
BUNDLED_EMPTY_SHA = "ab15e88be59c0ee93e63160c34b08485e3136d1f88313f24dcd67e482ecbed05"
REPIN_SHA = "1111111111111111111111111111111111111111111111111111111111111111"

PROJECT = """\
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
    a `nexus.registry.yaml` with one vehicle and no scene resolves its vehicle, `astro_max_base` and
    `empty`, and no warning prints.
    """
    (tmp_path / "nexus.registry.yaml").write_text(PROJECT)
    monkeypatch.chdir(tmp_path)

    with caplog.at_level(logging.WARNING):
        reg = load_registry()
        resolved = ("project_vehicle" in reg.vehicles, "astro_max_base" in reg.vehicles, "empty" in reg.scenes)

    assert (resolved, _warnings(caplog)) == ((True, True, True), [])


def test_each_entry_resolves_against_its_own_catalogs_base(tmp_path, monkeypatch):
    """Each entry resolves against its own catalog's base: a project entry under the project's
    `s3://` base, a bundled entry under the public CloudFront prefix.
    """
    (tmp_path / "nexus.registry.yaml").write_text(PROJECT)
    monkeypatch.chdir(tmp_path)

    reg = load_registry()

    assert (reg.vehicles["project_vehicle"].usd.url, reg.vehicles["astro_max_base"].usd.url) == (
        "s3://example-bucket/catalog/assets/usd/vehicles/project_vehicle-abc.usdz",
        BUNDLED_ASTRO,
    )


def test_a_name_in_both_catalogs_takes_the_projects_entry_with_a_warning(tmp_path, monkeypatch, caplog):
    """A name in both catalogs: the project's entry wins, with a warning. A project catalog that
    redefines `empty` with another hash resolves `empty` to the project's hash, and one warning names
    `empty` and both hashes.
    """
    (tmp_path / "nexus.registry.yaml").write_text(REPIN)
    monkeypatch.chdir(tmp_path)

    with caplog.at_level(logging.WARNING):
        sha = load_registry().scenes["empty"].usd.sha256
    named = [all(s in m for s in ("empty", REPIN_SHA, BUNDLED_EMPTY_SHA)) for m in _warnings(caplog)]

    assert (sha, named) == (REPIN_SHA, [True])


def test_a_catalog_named_by_path_also_extends_the_bundled_one(tmp_path, monkeypatch):
    """A catalog named by path also extends the bundled one: a file with one vehicle, loaded by
    `load_registry(path)`, resolves the bundled `astro_max_base` too.
    """
    named = tmp_path / "project.yaml"
    named.write_text(PROJECT)
    monkeypatch.chdir(tmp_path)

    assert load_registry(named).vehicles["astro_max_base"].usd.url == BUNDLED_ASTRO


def test_a_catalog_entry_that_still_carries_px4_fails_to_load(tmp_path):
    """A catalog entry that still carries `px4:` fails to load: the airframe lives in the vehicle's Universal Scene Description (USD) file.

    Given a project catalog whose vehicle entry has `px4: { airframe: astro_max }`, when
    `nexus.Registry.from_yaml(path)` loads it, then it raises and names the `px4` field.
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
        nexus.Registry.from_yaml(catalog)


MY_QUAD = """\
vehicles:
  my_quad:
    usd: { url: "file:///my_quad.usda", sha256: abc }
"""


def test_a_catalog_with_a_defaults_block_fails_to_load_and_says_to_name_the_vehicle_and_scene(tmp_path):
    """A project catalog with a `defaults` block fails to load, and the error names the removal and
    says to name the vehicle and scene on the run.

    Given a catalog with `defaults: { vehicle: my_quad }`, when `nexus.Registry.from_yaml(path)` reads
    it, then it raises an error that names `defaults` as removed and points at `--vehicle` and `--scene`.
    """
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(MY_QUAD + "defaults: { vehicle: my_quad }\n")

    with pytest.raises((RegistryError, ValueError)) as e:
        nexus.Registry.from_yaml(catalog)

    assert [w in str(e.value) for w in ("defaults", "remov", "--vehicle", "--scene")] == [True] * 4


def test_a_project_catalog_that_lists_only_what_it_adds_loads_on_its_own(tmp_path):
    """A project catalog that lists only what it adds loads on its own.

    Given a file with one vehicle, `scenes: {}` and no `defaults`, when `nexus.Registry.from_yaml(path)`
    reads it, then it returns a catalog with that one vehicle and no error.
    """
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(MY_QUAD + "scenes: {}\n")

    assert list(nexus.Registry.from_yaml(catalog).vehicles) == ["my_quad"]


BUNDLED_ASTRO_SHA = "38700d1d05b739bba445b1b1d4fb35c7aec8ec4ef05d49d4dcb8b8941ef84ca9"
BUNDLED_FPV_SHA = "51cec8c50db9b50d9a8baa86730b780e574d1d8846c5483ffe92d168cf2cbbe8"

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

    Given a project catalog with `vehicles: { astro_max_base: { usd: … } }`, when `load_registry`
    loads it, then `astro_max_base` resolves to the project's USD, `astro_max_fpv` still resolves to
    the bundled one, and one warning names both sha256 values.
    """
    (tmp_path / "nexus.registry.yaml").write_text(KEYED_REPIN)
    monkeypatch.chdir(tmp_path)

    with caplog.at_level(logging.WARNING):
        reg = load_registry()
    shas = (reg.vehicles["astro_max_base"].usd.sha256, reg.vehicles["astro_max_fpv"].usd.sha256)
    named = [all(s in m for s in ("astro_max_base", REPIN_SHA, BUNDLED_ASTRO_SHA)) for m in _warnings(caplog)]

    assert (shas, named) == ((REPIN_SHA, BUNDLED_FPV_SHA), [True])


def test_a_project_catalog_that_adds_one_vehicle_by_key_still_flies_the_bundled_ones(tmp_path, monkeypatch):
    """A project catalog that adds one vehicle by key still flies the bundled vehicles and scenes.

    Given a project catalog with only `vehicles: { project_vehicle: … }`, when `load_registry` loads
    it, then `project_vehicle`, `astro_max_base`, `astro_max_fpv` and the scene `empty` all resolve.
    """
    (tmp_path / "nexus.registry.yaml").write_text(KEYED_PROJECT)
    monkeypatch.chdir(tmp_path)

    reg = load_registry()
    resolved = [name in reg.vehicles for name in ("project_vehicle", "astro_max_base", "astro_max_fpv")]

    assert (resolved, "empty" in reg.scenes) == ([True, True, True], True)
