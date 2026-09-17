"""End-to-end resolution: name -> variant -> tested-config receipt, plus fetch."""

import hashlib
import logging
from pathlib import Path

from nexus._src.config import LaunchConfig, Registry, TestedConfig, resolve


def _reg(vehicle_usd):
    return Registry.from_dict(
        {
            "vehicles": [{"name": "black", "usd": vehicle_usd, "px4": {"airframe": "80001"}}],
            "scenes": {"empty": {}},
            "defaults": {"vehicle": "black", "scene": "empty"},
        }
    )


def test_resolve_no_fetch_builds_fully_specified_receipt():
    reg = _reg({"url": "https://x/astro.usdz", "sha256": "deadbeef"})
    rl = resolve(LaunchConfig().set_vehicle("black"), reg, fetch=False)
    tc = rl.tested_config
    # the named variant, fully specified
    assert tc.vehicle == "black"
    assert tc.px4.airframe == "80001"
    assert tc.vehicle_usd.sha256 == "deadbeef"
    assert tc.scene == "empty" and tc.scene_usd is None
    assert rl.vehicle_usd_path is None  # fetch off


def test_resolve_fetches_and_verifies_file_asset(tmp_path):
    # a real file:// asset, content-addressed by its true sha256, mirroring newton-assets' resolver test
    blob = tmp_path / "astro.usdz"
    blob.write_bytes(b"USD-PLACEHOLDER-BYTES")
    sha = hashlib.sha256(blob.read_bytes()).hexdigest()
    reg = _reg({"url": blob.as_uri(), "sha256": sha, "filename": "astro.usdz"})

    rl = resolve(LaunchConfig().set_vehicle("black"), reg, fetch=True, cache_dir=tmp_path / "cache")

    assert rl.vehicle_usd_path is not None
    assert Path(rl.vehicle_usd_path).read_bytes() == b"USD-PLACEHOLDER-BYTES"


def test_receipt_round_trips_json():
    reg = _reg({"url": "https://x/astro.usdz", "sha256": "deadbeef"})
    rl = resolve(LaunchConfig().set_vehicle("black"), reg, fetch=False)
    again = TestedConfig.model_validate_json(rl.tested_config.to_json())
    assert again == rl.tested_config


def test_receipt_carries_sensors_and_environment():
    reg = _reg({"url": "https://x/astro.usdz", "sha256": "deadbeef"})
    lc = LaunchConfig.from_dict(
        {
            "vehicle": "black",
            "sensors": {"imu": {"rate": 250}},
            "environment": {"wind": {"mean": [3, 0, 0]}},
        }
    )
    tc = resolve(lc, reg, fetch=False).tested_config
    assert tc.sensors == {"imu": {"rate": 250}}
    assert tc.environment is not None and tc.environment.wind == {"mean": [3, 0, 0]}


def _fpv_reg(vehicle_usd, scene_usd, **scene_extra):
    """A registry with a synthetic 'fpv' scene, a Universal Scene Description (USD) file plus a geo
    origin, which a launch names.
    """
    return Registry.from_dict(
        {
            "vehicles": [{"name": "black", "usd": vehicle_usd, "px4": {"airframe": "80001"}}],
            "scenes": {
                "empty": {},
                "fpv": {
                    "usd": scene_usd,
                    "geodetic_origin": {"lat": 37.5, "lon": -122.3},
                    **scene_extra,
                },
            },
            "defaults": {"vehicle": "black", "scene": "empty"},
        }
    )


def _file_asset(tmp_path, name, body):
    blob = tmp_path / name
    blob.write_bytes(body)
    return {"url": blob.as_uri(), "sha256": hashlib.sha256(blob.read_bytes()).hexdigest(), "filename": name}


def test_scene_surfaces_usd_path_and_geo(tmp_path):
    """A scene resolves with scene_usd_path populated *and* the geo origin surfaced so the
    environment can anchor. What the USD *is*, simulation geometry or the visual world, is the
    USD's own business: the launch glue reads its authored physics schemas, not the registry.
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
    starts, rides the receipt so the launch glue can hand it to the renderer; missing = None.
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
    sha256'd into the receipt the same way as a registry asset, with no ``start``, so the scene's own origin.
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


def test_default_scene_resolves_without_usd():
    """The default `empty` scene has no USD: nothing to route; flat ground."""
    reg = _reg({"url": "https://x/astro.usdz", "sha256": "deadbeef"})
    rl = resolve(LaunchConfig().set_vehicle("black"), reg, fetch=False)
    assert rl.tested_config.scene == "empty"
    assert rl.scene_usd_path is None


def test_a_named_vehicle_beats_the_default():
    reg = Registry.from_dict(
        {
            "vehicles": [
                {"name": "black", "usd": {"url": "file:///b", "sha256": "0"}, "px4": {"airframe": "80001"}},
                {"name": "blue", "usd": {"url": "file:///u", "sha256": "0"}, "px4": {"airframe": "80002"}},
            ],
            "scenes": {"empty": {}},
            "defaults": {"vehicle": "black", "scene": "empty"},
        }
    )
    assert resolve(LaunchConfig.from_dict({"vehicle": "blue"}), reg, fetch=False).tested_config.px4.airframe == "80002"


def test_a_vehicle_still_resolves_by_name():
    """A vehicle still resolves by name: a run that names a vehicle resolves it, and a run that names
    none takes the registry default.

    Against the shipped catalog, not a synthetic one, because the default is registry data. The test
    reads each run by its USD filename: the filename names the variant and survives an asset update.
    """
    named = resolve(LaunchConfig().set_vehicle("astro_max_base"), fetch=False)
    unnamed = resolve(LaunchConfig().set_vehicle(None), fetch=False)
    assert (named.tested_config.vehicle_usd.filename, unnamed.tested_config.vehicle_usd.filename) == (
        "astro_max_base.usdz",
        "astro_max_base.usdz",
    )


def _catalog(vehicle: str) -> str:
    """A one-vehicle registry naming *vehicle*, enough to tell two catalogs apart."""
    return (
        "vehicles:\n"
        f"  - name: {vehicle}\n"
        '    usd: { url: "file:///' + '{v}.usdz", sha256: "0" }\n'.replace("{v}", vehicle) + "scenes:\n  empty: {}\n"
        f"defaults: {{ vehicle: {vehicle}, scene: empty }}\n"
    )


def test_an_explicit_registry_beats_discovery(tmp_path, monkeypatch):
    """An explicit registry beats discovery: a run told which file to fly flies that one, whatever a
    walk up from the working directory would have found.
    """
    (tmp_path / "nexus.registry.yaml").write_text(_catalog("discovered"))
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(_catalog("explicit"))
    monkeypatch.chdir(tmp_path)

    rl = resolve(LaunchConfig(registry=str(explicit)), fetch=False)

    assert rl.tested_config.vehicle == "explicit"


def test_a_run_says_which_catalog_it_flew(tmp_path, monkeypatch, caplog):
    """A run says which catalog it flew: the registry's path is in the run's log and in its
    tested-config receipt.
    """
    registry = tmp_path / "nexus.registry.yaml"
    registry.write_text(_catalog("discovered"))
    monkeypatch.chdir(tmp_path)

    with caplog.at_level(logging.INFO):
        rl = resolve(LaunchConfig(), fetch=False)

    assert (rl.tested_config.registry, str(registry) in caplog.text) == (str(registry), True)
