"""End-to-end resolution: name -> variant -> tested-config receipt, plus fetch."""

import hashlib
import json
import logging
from pathlib import Path

import pytest

import nexus_sim
from nexus_sim._src.config import Catalog, LaunchConfig, NoMatchError, TestedConfig, resolve


def _reg(vehicle_usd):
    return Catalog.from_dict(
        {
            "vehicles": {"black": {"usd": vehicle_usd}},
            "scenes": {"empty": {}},
        }
    )


def test_resolve_no_fetch_builds_fully_specified_receipt():
    reg = _reg({"url": "https://x/astro.usdz", "sha256": "deadbeef"})
    rl = resolve(LaunchConfig().set_vehicle("black").set_scene("empty"), reg, fetch=False)
    tc = rl.tested_config
    # the named variant, fully specified
    assert tc.vehicle == "black"
    assert tc.vehicle_usd.sha256 == "deadbeef"
    assert tc.scene == "empty" and tc.scene_usd is None
    assert rl.vehicle_usd_path is None  # fetch off


def test_resolve_fetches_and_verifies_file_asset(tmp_path):
    # a real file:// asset, content-addressed by its true sha256, as the asset resolver's own tests do
    blob = tmp_path / "astro.usdz"
    blob.write_bytes(b"USD-PLACEHOLDER-BYTES")
    sha = hashlib.sha256(blob.read_bytes()).hexdigest()
    reg = _reg({"url": blob.as_uri(), "sha256": sha, "filename": "astro.usdz"})

    rl = resolve(LaunchConfig().set_vehicle("black").set_scene("empty"), reg, fetch=True, cache_dir=tmp_path / "cache")

    assert rl.vehicle_usd_path is not None
    assert Path(rl.vehicle_usd_path).read_bytes() == b"USD-PLACEHOLDER-BYTES"


def test_receipt_round_trips_json():
    reg = _reg({"url": "https://x/astro.usdz", "sha256": "deadbeef"})
    rl = resolve(LaunchConfig().set_vehicle("black").set_scene("empty"), reg, fetch=False)
    again = TestedConfig.model_validate_json(rl.tested_config.to_json())
    assert again == rl.tested_config


def _fpv_reg(vehicle_usd, scene_usd, **scene_extra):
    """A catalog with a synthetic 'fpv' scene, a Universal Scene Description (USD) file plus a geo
    origin, which a launch names.
    """
    return Catalog.from_dict(
        {
            "vehicles": {"black": {"usd": vehicle_usd}},
            "scenes": {
                "empty": {},
                "fpv": {
                    "usd": scene_usd,
                    "geodetic_origin": {"lat": 37.5, "lon": -122.3},
                    **scene_extra,
                },
            },
        }
    )


def _file_asset(tmp_path, name, body):
    blob = tmp_path / name
    blob.write_bytes(body)
    return {"url": blob.as_uri(), "sha256": hashlib.sha256(blob.read_bytes()).hexdigest(), "filename": name}


def test_scene_surfaces_usd_path_and_geo(tmp_path):
    """A scene resolves with scene_usd_path populated *and* the geo origin surfaced so the
    environment can anchor. What the USD *is*, simulation geometry or the visual world, is the
    USD's own business: the run's builder reads its authored physics schemas, not the catalog.
    """
    reg = _fpv_reg(
        _file_asset(tmp_path, "veh.usdz", b"USD-VEH"),
        _file_asset(tmp_path, "stage.usdz", b"USD-STAGE"),
    )
    lc = LaunchConfig().set_vehicle("black").set_scene("fpv")
    rl = resolve(lc, reg, fetch=True, cache_dir=tmp_path / "cache")

    assert rl.scene_usd_path is not None
    assert Path(rl.scene_usd_path).read_bytes() == b"USD-STAGE"  # carried for the consumer
    # geo origin surfaces: it must reach ConstantEnvironment.from_gps even though physics is empty
    geo = rl.tested_config.geodetic_origin
    assert geo is not None and (geo.lat, geo.lon) == (37.5, -122.3)


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


def test_the_empty_scene_resolves_without_usd():
    """The `empty` scene has no USD: nothing to route; flat ground."""
    reg = _reg({"url": "https://x/astro.usdz", "sha256": "deadbeef"})
    rl = resolve(LaunchConfig().set_vehicle("black").set_scene("empty"), reg, fetch=False)
    assert rl.tested_config.scene == "empty"
    assert rl.scene_usd_path is None


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
        '    usd: { url: "file:///' + '{v}.usdz", sha256: "0" }\n'.replace("{v}", vehicle) + "scenes:\n  empty: {}\n"
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

    assert (rl.tested_config.catalog, str(catalog) in caplog.text) == (str(catalog), True)


def test_a_launch_that_names_a_dropped_variant_fails_before_it_flies(tmp_path, monkeypatch):
    """A launch that names a dropped variant fails before it flies: the shipped catalog no longer
    carries `astro_max_fpv_lr1`, so the launch stops and says no vehicle has that name.
    """
    monkeypatch.chdir(tmp_path)  # no nexus.catalog.yaml beside the run
    with pytest.raises(NoMatchError, match="no vehicle named 'astro_max_fpv_lr1'"):
        with nexus_sim.Sim(vehicle="astro_max_fpv_lr1", scene="empty"):
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
