"""Registry resolution: name lookup, load-time validation, and asset-reference expansion."""

import logging

import pytest

import nexus
from nexus._src.config import NoMatchError, Registry, RegistryError, load_registry


def _veh(name, airframe):
    return {"name": name, "usd": {"url": f"file:///{name}", "sha256": "0"}, "px4": {"airframe": airframe}}


BASE = _veh("base", "80001")
FPV = _veh("fpv", "80002")
FPV_LR1 = _veh("fpv_lr1", "80003")


def _reg(vehicles, **kw):
    return Registry.from_dict({"vehicles": vehicles, "scenes": {"empty": {}}, **kw})


def test_by_name():
    reg = _reg([BASE, FPV_LR1])
    assert reg.by_name("fpv_lr1").px4.airframe == "80003"


def test_unknown_name_raises():
    reg = _reg([BASE])
    with pytest.raises(NoMatchError):
        reg.by_name("nope")


def test_duplicate_name_errors_at_load():
    with pytest.raises(RegistryError):
        _reg([FPV, dict(FPV, usd={"url": "file:///dup", "sha256": "0"})])  # two variants named 'fpv'


def test_dangling_default_scene_errors_at_load():
    with pytest.raises(RegistryError):
        Registry.from_dict({"vehicles": [BASE], "scenes": {"empty": {}}, "defaults": {"scene": "nope"}})


def test_dangling_default_vehicle_errors_at_load():
    with pytest.raises(RegistryError):
        _reg([BASE], defaults={"vehicle": "nope"})


SIBLING = """\
vehicles:
  - name: sibling_vehicle
    usd: { url: "file:///sibling.usdz", sha256: "0" }
scenes:
  empty: {}
defaults: { vehicle: sibling_vehicle, scene: empty }
"""


def test_a_registry_beside_the_run_wins_over_the_one_in_the_wheel(tmp_path, monkeypatch):
    """A registry beside the run wins over the one in the wheel: `nexus.registry.yaml` in the working
    directory or a parent is the catalog a run flies, and where no parent holds one the run flies the
    catalog the wheel ships.
    """
    project = tmp_path / "sibling"
    (project / "scripts").mkdir(parents=True)
    (project / "nexus.registry.yaml").write_text(SIBLING)

    monkeypatch.chdir(project / "scripts")  # started below the file, which the walk up still finds
    beside = [v.name for v in load_registry().vehicles]
    monkeypatch.chdir(tmp_path)  # no parent of this one holds a registry
    shipped = [v.name for v in load_registry().vehicles]

    assert (beside, "astro_max_base" in shipped) == (["sibling_vehicle"], True)


def test_an_entry_addresses_its_own_blob():
    """An entry addresses its own blob: a full URL fetches from where it says, and `{name, sha256}`
    resolves against its own registry's base.
    """
    reg = Registry.from_dict(
        {
            "assets": {"base": "https://assets.example/catalog"},
            "vehicles": [
                {"name": "derived", "usd": {"name": "astro", "sha256": "abc"}},
                {"name": "direct", "usd": {"url": "s3://example-bucket/elsewhere/x.usdz", "sha256": "def"}},
            ],
            "scenes": {"empty": {}},
            "defaults": {"vehicle": "derived", "scene": "empty"},
        }
    )
    assert (reg.by_name("derived").usd.url, reg.by_name("direct").usd.url) == (
        "https://assets.example/catalog/assets/usd/vehicles/astro-abc.usdz",
        "s3://example-bucket/elsewhere/x.usdz",
    )


def test_the_shipped_catalog_resolves_exactly_as_today():
    """The shipped catalog resolves exactly as today: every vehicle and scene in the framework's own
    registry keeps the URL it has now.
    """
    reg = load_registry()
    base = "https://d2837jz4fvtxko.cloudfront.net/public/assets/usd"
    hosted = [v.usd.url for v in reg.vehicles] + [s.usd.url for s in reg.scenes.values() if s.usd]
    astro = reg.by_name("astro_max_base").usd
    assert all(u.startswith(base) for u in hosted)
    assert astro.url == f"{base}/vehicles/astro_max_base-{astro.sha256}.usdz"


def test_the_shipped_catalog_names_two_vehicles():
    """The shipped catalog names two vehicles, `astro_max_base` and `astro_max_fpv`: the payload
    variants leave the tree, and a project that wants one brings it in its own catalog.
    """
    assert [v.name for v in nexus.Registry.from_yaml().vehicles] == ["astro_max_base", "astro_max_fpv"]


HOSTED = "https://d2837jz4fvtxko.cloudfront.net/public/assets/usd/vehicles"


def test_the_two_astro_max_vehicles_keep_the_usds_they_fly_today():
    """`astro_max_base` and `astro_max_fpv` keep the USDs they fly today: the published USDs, pinned
    by hash, stay the source of record once the scripts that authored them leave the tree.
    """
    reg = nexus.Registry.from_yaml()
    assert [reg.by_name(name).usd.url for name in ("astro_max_base", "astro_max_fpv")] == [
        f"{HOSTED}/astro_max_base-d2f538aaeeef4c2a951cac3f9e062003e7c4cc6e5e78b2960076b69a393f92e8.usdz",
        f"{HOSTED}/astro_max_fpv-9dc55c51a6912faacf2614a7b3e244bf10233f440a8dd920490857df03daf5ad.usdz",
    ]


BUNDLED_ASTRO = f"{HOSTED}/astro_max_base-d2f538aaeeef4c2a951cac3f9e062003e7c4cc6e5e78b2960076b69a393f92e8.usdz"
BUNDLED_EMPTY_SHA = "ab15e88be59c0ee93e63160c34b08485e3136d1f88313f24dcd67e482ecbed05"
REPIN_SHA = "1111111111111111111111111111111111111111111111111111111111111111"

PROJECT = """\
assets:
  base: s3://example-bucket/catalog
vehicles:
  - name: project_vehicle
    usd: { name: project_vehicle, sha256: abc }
defaults: { vehicle: project_vehicle }
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
        resolved = (reg.by_name("project_vehicle").name, reg.by_name("astro_max_base").name, "empty" in reg.scenes)

    assert (resolved, _warnings(caplog)) == (("project_vehicle", "astro_max_base", True), [])


def test_each_entry_resolves_against_its_own_catalogs_base(tmp_path, monkeypatch):
    """Each entry resolves against its own catalog's base: a project entry under the project's
    `s3://` base, a bundled entry under the public CloudFront prefix.
    """
    (tmp_path / "nexus.registry.yaml").write_text(PROJECT)
    monkeypatch.chdir(tmp_path)

    reg = load_registry()

    assert (reg.by_name("project_vehicle").usd.url, reg.by_name("astro_max_base").usd.url) == (
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

    assert load_registry(named).by_name("astro_max_base").usd.url == BUNDLED_ASTRO
