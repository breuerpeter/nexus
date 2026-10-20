"""End-to-end resolution: name -> variant -> tested-config receipt, plus fetch."""

import hashlib
import json
import logging
from pathlib import Path

import pytest

import nexus_uav
from nexus_uav._src.config import Catalog, LaunchConfig, NoMatchError, TestedConfig, resolve

# A scene's file, which a test that resolves with no fetch never reads.
EMPTY = {"url": "file:///empty.usdz", "sha256": "0"}


def _reg(vehicle_usd):
    return Catalog.from_dict(
        {
            "vehicles": {"black": {"usd": vehicle_usd}},
            "scenes": {"empty": {"usd": EMPTY}},
        }
    )


def test_resolve_no_fetch_builds_fully_specified_receipt():
    reg = _reg({"url": "https://x/astro.usdz", "sha256": "deadbeef"})
    rl = resolve(LaunchConfig().set_vehicle("black").set_scene("empty"), reg, fetch=False)
    tc = rl.tested_config
    # the named variant, fully specified
    assert tc.vehicle == "black"
    assert tc.vehicle_usd.sha256 == "deadbeef"
    assert tc.scene == "empty" and tc.scene_usd.url == EMPTY["url"]
    assert rl.vehicle_usd_path is None  # fetch off


def test_resolve_fetches_and_verifies_file_asset(tmp_path):
    # a real file:// asset, content-addressed by its true sha256, as the asset resolver's own tests do
    blob = tmp_path / "astro.usdz"
    blob.write_bytes(b"USD-PLACEHOLDER-BYTES")
    sha = hashlib.sha256(blob.read_bytes()).hexdigest()
    reg = _fpv_reg(
        {"url": blob.as_uri(), "sha256": sha, "filename": "astro.usdz"},
        _file_asset(tmp_path, "stage.usdz", b"USD-STAGE"),
    )

    rl = resolve(LaunchConfig().set_vehicle("black").set_scene("fpv"), reg, fetch=True, cache_dir=tmp_path / "cache")

    assert rl.vehicle_usd_path is not None
    assert Path(rl.vehicle_usd_path).read_bytes() == b"USD-PLACEHOLDER-BYTES"


def test_receipt_round_trips_json():
    reg = _reg({"url": "https://x/astro.usdz", "sha256": "deadbeef"})
    rl = resolve(LaunchConfig().set_vehicle("black").set_scene("empty"), reg, fetch=False)
    again = TestedConfig.model_validate_json(rl.tested_config.to_json())
    assert again == rl.tested_config


def _fpv_reg(vehicle_usd, scene_usd, **scene_extra):
    """A catalog with a synthetic 'fpv' scene, a Universal Scene Description (USD) file, which a launch names."""
    return Catalog.from_dict(
        {
            "vehicles": {"black": {"usd": vehicle_usd}},
            "scenes": {
                "empty": {"usd": EMPTY},
                "fpv": {
                    "usd": scene_usd,
                    **scene_extra,
                },
            },
        }
    )


def _file_asset(tmp_path, name, body):
    blob = tmp_path / name
    blob.write_bytes(body)
    return {"url": blob.as_uri(), "sha256": hashlib.sha256(blob.read_bytes()).hexdigest(), "filename": name}


def test_scene_surfaces_usd_path(tmp_path):
    """A scene resolves with scene_usd_path populated. What the USD *is*, simulation geometry or the
    visual world, and where it sits on Earth, is the USD's own business: the run's builder reads its
    authored schemas, not the catalog.
    """
    reg = _fpv_reg(
        _file_asset(tmp_path, "veh.usdz", b"USD-VEH"),
        _file_asset(tmp_path, "stage.usdz", b"USD-STAGE"),
    )
    lc = LaunchConfig().set_vehicle("black").set_scene("fpv")
    rl = resolve(lc, reg, fetch=True, cache_dir=tmp_path / "cache")

    assert rl.scene_usd_path is not None
    assert Path(rl.scene_usd_path).read_bytes() == b"USD-STAGE"  # carried for the consumer


def test_scene_start_surfaces_on_the_receipt(tmp_path):
    """The scene's ``start``, the scene-frame point placed at the world origin where the drone
    starts, rides the receipt so the run's builder can hand it to the renderer; missing = None.
    """
    reg = _fpv_reg(
        _file_asset(tmp_path, "veh.usdz", b"USD-VEH"),
        _file_asset(tmp_path, "stage.usdz", b"USD-STAGE"),
        start=[-38.0, -22.0, 1.4],
    )
    lc = LaunchConfig().set_vehicle("black").set_scene("fpv")
    rl = resolve(lc, reg, fetch=True, cache_dir=tmp_path / "cache")
    assert rl.tested_config.scene_start == (-38.0, -22.0, 1.4)

    no_start = _fpv_reg(
        _file_asset(tmp_path, "veh2.usdz", b"USD-VEH"),
        _file_asset(tmp_path, "stage2.usdz", b"USD-STAGE"),
    )
    assert resolve(lc, no_start, fetch=False).tested_config.scene_start is None


def test_local_scene_usd_path(tmp_path):
    """``--scene <path.usd>`` flies an unregistered converted scene: it uses the file in place,
    sha256'd into the receipt the same way as a catalog asset, with no ``start``, so the scene's own origin.
    """
    reg = _fpv_reg(
        _file_asset(tmp_path, "veh.usdz", b"USD-VEH"),
        _file_asset(tmp_path, "stage.usdz", b"USD-STAGE"),
    )
    scn = tmp_path / "site_scan.usd"
    scn.write_bytes(b"USD-LOCAL-SCENE")
    lc = LaunchConfig().set_vehicle("black").set_scene(str(scn))
    rl = resolve(lc, reg, fetch=True, cache_dir=tmp_path / "cache")
    assert rl.tested_config.scene_start is None
    assert Path(rl.scene_usd_path).read_bytes() == b"USD-LOCAL-SCENE"


def test_a_vehicle_still_resolves_by_name():
    """A vehicle still resolves by name: a run that names a vehicle of the shipped catalog resolves it.

    The test reads the run by its USD filename: the filename names the variant and survives an asset update.
    """
    named = resolve(LaunchConfig().set_vehicle("astro_max_base").set_scene("empty"), fetch=False)
    assert named.tested_config.vehicle_usd.filename == "astro_max_base.usdz"


def _catalog(vehicle: str) -> str:
    """A one-vehicle catalog naming *vehicle*, enough to tell two catalogs apart."""
    return (
        "vehicles:\n"
        f"  {vehicle}:\n"
        '    usd: { url: "file:///'
        + '{v}.usdz", sha256: "0" }\n'.replace("{v}", vehicle)
        + 'scenes:\n  empty:\n    usd: { url: "file:///empty.usdz", sha256: "0" }\n'
    )


def test_an_explicit_catalog_beats_discovery(tmp_path, monkeypatch):
    """An explicit catalog beats discovery: a run told which file to fly flies that one, whatever a
    walk up from the working directory would have found.
    """
    (tmp_path / "nexus.catalog.yaml").write_text(_catalog("discovered"))
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(_catalog("explicit"))
    monkeypatch.chdir(tmp_path)

    rl = resolve(LaunchConfig(catalog=str(explicit), vehicle="explicit", scene="empty"), fetch=False)

    assert rl.tested_config.vehicle == "explicit"


def test_a_run_says_which_catalog_it_flew(tmp_path, monkeypatch, caplog):
    """A run says which catalog it flew: the catalog's path is in the run's log and in its
    tested-config receipt.
    """
    catalog = tmp_path / "nexus.catalog.yaml"
    catalog.write_text(_catalog("discovered"))
    monkeypatch.chdir(tmp_path)

    with caplog.at_level(logging.INFO):
        rl = resolve(LaunchConfig(vehicle="discovered", scene="empty"), fetch=False)

    assert (rl.tested_config.catalogs[-1].path, str(catalog) in caplog.text) == (catalog, True)


def test_a_launch_that_names_a_dropped_variant_fails_before_it_flies(tmp_path, monkeypatch):
    """A launch that names a dropped variant fails before it flies: the shipped catalog no longer
    carries `astro_max_fpv_lr1`, so the launch stops and says no vehicle has that name.
    """
    monkeypatch.chdir(tmp_path)  # no nexus.catalog.yaml beside the run
    with pytest.raises(NoMatchError, match="no vehicle named 'astro_max_fpv_lr1'"):
        with nexus_uav.Sim(vehicle="astro_max_fpv_lr1", scene="empty"):
            pass


def test_the_receipt_carries_no_environment_field():
    """The `environment` launch key and the receipt's `environment` field go, since nothing reads them."""
    reg = _reg({"url": "https://x/astro.usdz", "sha256": "deadbeef"})
    rl = resolve(LaunchConfig().set_vehicle("black").set_scene("empty"), reg, fetch=False)

    assert "environment" not in json.loads(rl.tested_config.to_json())


def test_a_catalog_that_keys_vehicles_by_name_flies_a_vehicle_by_its_key(tmp_path, monkeypatch):
    """A catalog that keys `vehicles` by name loads, and a run flies a vehicle by its key.

    Given a project catalog with `vehicles: { my_quad: { usd: … } }` and no `name` field, when a
    launch names `--vehicle my_quad` and resolves, then it resolves that entry's USD and the receipt
    records the vehicle `my_quad`.
    """
    (tmp_path / "nexus.catalog.yaml").write_text(
        'vehicles:\n  my_quad:\n    usd: { url: "file:///my_quad.usdz", sha256: abc }\n'
    )
    monkeypatch.chdir(tmp_path)

    tc = resolve(LaunchConfig().set_vehicle("my_quad").set_scene("empty"), fetch=False).tested_config

    assert (tc.vehicle, tc.vehicle_usd.url) == ("my_quad", "file:///my_quad.usdz")


def test_a_launch_that_names_an_unknown_vehicle_fails_and_lists_the_catalogs_vehicle_names(tmp_path, monkeypatch):
    """A launch that names an unknown vehicle fails and lists the catalog's vehicle names.

    Given the bundled catalog, when a launch names `--vehicle nope`, then it raises `NoMatchError`
    whose message lists `astro_max_base` and `astro_max_fpv`.
    """
    monkeypatch.chdir(tmp_path)  # no parent of this one holds a catalog

    with pytest.raises(NoMatchError) as e:
        resolve(LaunchConfig().set_vehicle("nope").set_scene("empty"), fetch=False)

    assert [name in str(e.value) for name in ("astro_max_base", "astro_max_fpv")] == [True, True]


def test_a_layered_scene_opens_with_the_scenes_units_up_axis_and_render_recipe(tmp_path):
    """A layered scene opens with the scene's units, up axis and render recipe.

    Given a local scene in metres, Z-up, with a render recipe in its layer data, and an override layer,
    when the run resolves, then the stage it opens for the scene has the scene's metres, Z axis and recipe.
    """
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    scene = tmp_path / "scene.usda"
    scene.write_text(
        "#usda 1.0\n(\n    customLayerData = {\n        dictionary renderSettings = {\n"
        '            bool "rtx:post:lensFlares:enabled" = 1\n        }\n    }\n'
        '    defaultPrim = "World"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\ndef Xform "World"\n{\n}\n'
    )
    layer = tmp_path / "layer.usda"
    layer.write_text("#usda 1.0\n")
    reg = _fpv_reg(_file_asset(tmp_path, "veh.usdz", b"USD-VEH"), _file_asset(tmp_path, "stage.usdz", b"USD-STAGE"))
    launch = LaunchConfig().set_vehicle("black").set_scene(str(scene))
    launch.layer = str(layer)
    stage = Usd.Stage.Open(resolve(launch, reg, fetch=False, cache_dir=tmp_path / "cache").scene_usd_path)

    assert (
        UsdGeom.GetStageMetersPerUnit(stage),
        UsdGeom.GetStageUpAxis(stage),
        dict(stage.GetRootLayer().customLayerData.get("renderSettings", {})),
    ) == (1.0, "Z", {"rtx:post:lensFlares:enabled": True})


# A catalog vehicle entry composes a base vehicle and its layers, #331. Each test writes small real USD
# files and serves them by `file://` URL, so the run opens the composed stage as it would a hosted one.

_BASE_USD = (
    "#usda 1.0\n"
    '(\n    defaultPrim = "veh"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\n'
    'def Xform "veh"\n{\n    def Xform "body"\n    {\n        custom int probe = 0\n    }\n}\n'
)


def _probe_layer(value: int, prim: str | None = None) -> str:
    """A layer that sets `/veh/body.probe` to *value*, and defines a camera *prim* under the body when named."""
    camera = (
        f'        def Camera "{prim}" (\n            apiSchemas = ["NexusCameraAPI"]\n        )\n'
        "        {\n            int nexus:width = 640\n            int nexus:height = 480\n"
        "            float nexus:rate = 10\n        }\n"
        if prim
        else ""
    )
    return f'#usda 1.0\n\nover "veh"\n{{\n    over "body"\n    {{\n        custom int probe = {value}\n{camera}    }}\n}}\n'


def _usd_asset(tmp_path, name, text):
    return _file_asset(tmp_path, name, text.encode())


def _composed_catalog(tmp_path, **vehicles):
    """A catalog whose vehicles are *vehicles*, each an entry dict, with one scene, `empty`, a real USD file."""
    scene = _usd_asset(
        tmp_path, "scene.usda", '#usda 1.0\n(\n    defaultPrim = "World"\n)\n\ndef Xform "World"\n{\n}\n'
    )
    return Catalog.from_dict({"vehicles": vehicles, "scenes": {"empty": {"usd": scene}}})


def _stage(tmp_path, catalog, vehicle, layer=None):
    """The vehicle stage a run of *vehicle* opens, resolved with fetch into a cache under *tmp_path*."""
    from pxr import Usd

    launch = LaunchConfig().set_vehicle(vehicle).set_scene("empty")
    launch.layer = layer
    resolved = resolve(launch, catalog, fetch=True, cache_dir=tmp_path / "cache")
    return Usd.Stage.Open(resolved.vehicle_usd_path), resolved.tested_config


def test_a_vehicle_entry_names_a_base_and_layers_and_a_run_of_it_flies_the_composition(tmp_path):
    """A vehicle entry names a base by its catalog name and layers by `{name, sha256}`, and a run of it
    flies the composition.

    Given a catalog with a base vehicle and an entry naming it plus a layer that adds a camera prim, when a
    run resolves the entry with fetch, then the vehicle USD the run opens holds the camera with its applied
    schema and every prim of the base.
    """
    pytest.importorskip("pxr")
    catalog = _composed_catalog(
        tmp_path,
        base={"usd": _usd_asset(tmp_path, "base.usda", _BASE_USD)},
        fpv={"base": "base", "layers": [_usd_asset(tmp_path, "camera.usda", _probe_layer(0, "cam"))]},
    )

    stage, _ = _stage(tmp_path, catalog, "fpv")
    cam = stage.GetPrimAtPath("/veh/body/cam")

    assert (
        cam.GetTypeName(),
        "NexusCameraAPI" in cam.GetAppliedSchemas(),
        stage.GetPrimAtPath("/veh/body").IsValid(),
    ) == ("Camera", True, True)


def test_the_first_listed_layer_is_strongest(tmp_path):
    """The first listed layer is strongest.

    Given an entry with two layers that author one attribute to different values, when the run resolves,
    then the stage reads the first layer's value.
    """
    pytest.importorskip("pxr")
    catalog = _composed_catalog(
        tmp_path,
        base={"usd": _usd_asset(tmp_path, "base.usda", _BASE_USD)},
        stacked={
            "base": "base",
            "layers": [
                _usd_asset(tmp_path, "first.usda", _probe_layer(1)),
                _usd_asset(tmp_path, "second.usda", _probe_layer(2)),
            ],
        },
    )

    stage, _ = _stage(tmp_path, catalog, "stacked")

    assert stage.GetAttributeAtPath("/veh/body.probe").Get() == 1


def test_the_receipt_records_the_base_and_each_layer(tmp_path):
    """The receipt records the base and each layer.

    Given a composed entry, when the run resolves, then the receipt carries the base's URL and sha256 and each
    layer's in the entry's order, and `--vehicle <path>` and a flat entry record as today.
    """
    base = _usd_asset(tmp_path, "base.usda", _BASE_USD)
    first = _usd_asset(tmp_path, "first.usda", _probe_layer(1))
    second = _usd_asset(tmp_path, "second.usda", _probe_layer(2))
    catalog = _composed_catalog(tmp_path, base={"usd": base}, stacked={"base": "base", "layers": [first, second]})
    local = tmp_path / "local.usda"
    local.write_text(_BASE_USD)

    def receipt(vehicle):
        return resolve(LaunchConfig().set_vehicle(vehicle).set_scene("empty"), catalog, fetch=False).tested_config

    composed, flat, path = receipt("stacked"), receipt("base"), receipt(str(local))

    assert (
        (composed.vehicle_usd.url, composed.vehicle_usd.sha256),
        [(layer.url, layer.sha256) for layer in composed.vehicle_layers],
        (flat.vehicle_usd.url, flat.vehicle_layers),
        (path.vehicle_usd.url, path.vehicle_layers),
    ) == (
        (base["url"], base["sha256"]),
        [(first["url"], first["sha256"]), (second["url"], second["sha256"])],
        (base["url"], []),
        (local.as_uri(), []),
    )


def test_a_launch_layer_composes_over_the_composed_entry(tmp_path):
    """A launch `--layer` composes over the composed entry.

    Given a composed entry whose layer authors a value and a launch layer that authors it too, when the run
    resolves, then the stage reads the launch layer's value and the receipt records the launch layer beside
    the entry's.
    """
    pytest.importorskip("pxr")
    entry_layer = _usd_asset(tmp_path, "entry.usda", _probe_layer(1))
    catalog = _composed_catalog(
        tmp_path,
        base={"usd": _usd_asset(tmp_path, "base.usda", _BASE_USD)},
        stacked={"base": "base", "layers": [entry_layer]},
    )
    launch_layer = tmp_path / "launch.usda"
    launch_layer.write_text(_probe_layer(9))

    stage, receipt = _stage(tmp_path, catalog, "stacked", layer=str(launch_layer))

    assert (
        stage.GetAttributeAtPath("/veh/body.probe").Get(),
        receipt.layer.sha256,
        [layer.sha256 for layer in receipt.vehicle_layers],
    ) == (9, hashlib.sha256(launch_layer.read_bytes()).hexdigest(), [entry_layer["sha256"]])


def test_a_base_may_itself_be_a_composed_entry(tmp_path):
    """A base can itself be a composed entry.

    Given entry B naming composed entry A as its base plus a layer of its own, when a run of B resolves, then
    the stage holds A's layer's prim and B's, and B's layer wins where both author one value.
    """
    pytest.importorskip("pxr")
    catalog = _composed_catalog(
        tmp_path,
        base={"usd": _usd_asset(tmp_path, "base.usda", _BASE_USD)},
        a={"base": "base", "layers": [_usd_asset(tmp_path, "a.usda", _probe_layer(1, "cam_a"))]},
        b={"base": "a", "layers": [_usd_asset(tmp_path, "b.usda", _probe_layer(2, "cam_b"))]},
    )

    stage, _ = _stage(tmp_path, catalog, "b")

    assert (
        stage.GetPrimAtPath("/veh/body/cam_a").IsValid(),
        stage.GetPrimAtPath("/veh/body/cam_b").IsValid(),
        stage.GetAttributeAtPath("/veh/body.probe").Get(),
    ) == (True, True, 2)


BUNDLED_ASTRO_BASE = (
    "https://d2837jz4fvtxko.cloudfront.net/public/assets/usd/vehicles/"
    "astro_max_base-f3590c8710e2d3dbe19a9b9fb9fae9b927001d91683616684a0cce93eec28e44.usdz"
)

PROJECT_OVER_BUNDLED = """\
assets:
  base: s3://example-bucket/catalog
vehicles:
  payload_vehicle:
    base: astro_max_base
    layers:
      - { name: payload, sha256: abc }
"""


def test_a_projects_entry_names_a_bundled_vehicle_as_its_base(tmp_path, monkeypatch):
    """A project's entry names a bundled vehicle as its base.

    Given a project catalog with its own `assets.base` whose entry names a bundled vehicle as base plus a
    compact layer, when the run resolves, then the base fetches from the bundled catalog's URL and the layer
    from the project's.
    """
    (tmp_path / "nexus.catalog.yaml").write_text(PROJECT_OVER_BUNDLED)
    monkeypatch.chdir(tmp_path)

    tc = resolve(LaunchConfig().set_vehicle("payload_vehicle").set_scene("empty"), fetch=False).tested_config

    assert (tc.vehicle_usd.url, [layer.url for layer in tc.vehicle_layers]) == (
        BUNDLED_ASTRO_BASE,
        ["s3://example-bucket/catalog/assets/usd/layers/payload-abc.usdz"],
    )


def test_a_composed_stage_hashes_the_same_from_any_cache(tmp_path):
    """A composed stage hashes the same from any cache, and differently from its base alone.

    Given one composed entry resolved into two caches, when each root hashes, then the two hashes are equal,
    and the base's file alone hashes to another value.
    """
    pytest.importorskip("pxr")
    from nexus_uav._src.assets import stage_hash

    catalog = _composed_catalog(
        tmp_path,
        base={"usd": _usd_asset(tmp_path, "base.usda", _BASE_USD)},
        fpv={"base": "base", "layers": [_usd_asset(tmp_path, "camera.usda", _probe_layer(0, "cam"))]},
    )
    launch = LaunchConfig().set_vehicle("fpv").set_scene("empty")
    roots = [resolve(launch, catalog, fetch=True, cache_dir=tmp_path / f"cache{i}").vehicle_usd_path for i in (1, 2)]

    hashes = [stage_hash(root) for root in roots]

    assert (hashes[0] == hashes[1], hashes[0] == stage_hash(tmp_path / "base.usda")) == (True, False)


def test_a_layer_whose_bytes_mismatch_its_sha256_fails_like_a_vehicles(tmp_path):
    """A layer whose bytes mismatch its sha256 fails as a vehicle's does.

    Given a composed entry whose layer's hosted bytes differ from its sha256, when the run resolves with
    fetch, then the fetch fails and names the layer.
    """
    layer = {**_usd_asset(tmp_path, "layer.usda", _probe_layer(1)), "sha256": "0" * 64}
    catalog = _composed_catalog(
        tmp_path,
        base={"usd": _usd_asset(tmp_path, "base.usda", _BASE_USD)},
        stacked={"base": "base", "layers": [layer]},
    )
    launch = LaunchConfig().set_vehicle("stacked").set_scene("empty")

    with pytest.raises(ValueError, match="hash mismatch") as e:
        resolve(launch, catalog, fetch=True, cache_dir=tmp_path / "cache")

    assert layer["url"] in str(e.value)
