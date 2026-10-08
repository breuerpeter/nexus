"""The run's builder: resolve a vehicle Universal Scene Description (USD) via nexus_sim._src.config + route by
control kind.

Exercises resolution, the vehicle's USD and the routing, not the heavy NewtonPhysics build,
which needs a real newton-loadable USD asset. The resolved USD is a file:// asset, sha-verified by
the real asset resolver.
"""

import hashlib
import logging
import re
from pathlib import Path

import pytest

from nexus_sim._src.config import Catalog, LaunchConfig
from nexus_sim._src.peers.px4_sitl.fake import Px4Fake
from tests.usd import sensor_vehicle as sv


@pytest.fixture(autouse=True)
def daemon(monkeypatch, tmp_path):
    """``build_from_launch`` starts the PX4 Software In The Loop (SITL) peer, which builds and runs PX4
    in docker. These tests exercise resolution and routing, so the docker daemon is a stand-in
    that records what the build asks of it, and ``$PX4_DIR`` a stand-in tree, so nothing fetches.
    """
    from docker.errors import NotFound

    import nexus_sim._src.peers.containers as containers

    runs = []

    class Images:
        def get(self, tag):
            return object()

    class Containers:
        def run(self, image, **kwargs):
            runs.append({"image": image, **kwargs})
            return b"" if not kwargs.get("detach", True) else object()

        def get(self, name):
            raise NotFound(name)

        def list(self, **kwargs):
            return []

    class Client:
        images = Images()
        containers = Containers()

    c = Client()
    monkeypatch.setattr(containers, "client", lambda: c)
    monkeypatch.setattr(containers, "_pump_logs", lambda container, log_path: None)
    px4 = tmp_path / "px4"
    px4.mkdir()
    (px4 / "Makefile").write_text("px4_sitl:\n")
    monkeypatch.setenv("PX4_DIR", str(px4))
    return runs


# A vehicle that declares PX4 and the PX4 SITL peer on its root prim, and nothing else.
PX4_VEHICLE = (
    b'#usda 1.0\n(\n    defaultPrim = "vehicle"\n)\n\n'
    b'def Xform "vehicle" (\n    prepend apiSchemas = ["NexusPx4API", "NexusPx4SitlAPI"]\n)\n'
    b'{\n    string nexus:airframe = "80001"\n}\n'
)


def _astro_catalog(usd_ref: dict) -> Catalog:
    return Catalog.from_dict(
        {
            "vehicles": {"astro": {"usd": usd_ref}},
            "scenes": {"empty": {}},
        }
    )


def _usd_ref(tmp_path) -> dict:
    blob = tmp_path / "vehicle.usda"
    blob.write_bytes(PX4_VEHICLE)
    return {"url": blob.as_uri(), "sha256": hashlib.sha256(blob.read_bytes()).hexdigest(), "filename": "vehicle.usda"}


def test_resolve_vehicle_usd_uses_resolved_usd(tmp_path):
    from nexus_sim._src.build.launch import resolve_vehicle_usd

    ref = _usd_ref(tmp_path)
    reg = _astro_catalog(ref)
    vehicle_usd, resolved = resolve_vehicle_usd(
        LaunchConfig().set_vehicle("astro").set_scene("empty"), reg, cache_dir=tmp_path / "cache"
    )
    # the vehicle's USD points at the verified, content-addressed local copy of the asset
    assert Path(vehicle_usd.cfg["usd_path"]).read_bytes() == PX4_VEHICLE
    assert ref["sha256"] in str(vehicle_usd.cfg["usd_path"])  # content-addressed cache path
    assert resolved.tested_config.px4.airframe == "80001"


def test_scenario_from_receipt_honors_dt_seed_device(tmp_path):
    from nexus_sim._src.build.launch import _scenario_from_receipt
    from nexus_sim._src.config import Runtime

    cfg = _scenario_from_receipt(Runtime(dt=0.01, seed=7, device="cpu"))
    assert cfg["physics"]["dt"] == 0.01
    assert cfg["physics"]["force_cpu"] is True
    assert cfg["seed"] == 7

    cfg_gpu = _scenario_from_receipt(Runtime(device="cuda"))
    assert cfg_gpu["physics"]["force_cpu"] is False


def _scene_usd(tmp_path, name, body: bytes) -> dict:
    blob = tmp_path / name
    blob.write_bytes(body)
    return {"url": blob.as_uri(), "sha256": hashlib.sha256(blob.read_bytes()).hexdigest(), "filename": name}


def test_scene_threads_uniformly_and_anchors_gps(tmp_path, monkeypatch):
    """A scene is *data*: its USD + start thread into the cfg for the one uniform ingestion, where
    physics parses the UsdPhysics prims and a renderer shows everything, with no routing anywhere,
    and its geodetic origin anchors *both* the render world, cfg.rtx.georef, *and* the
    Hardware In The Loop (HIL) Global Positioning System (GPS), cfg.sensors.gps.init, so the
    Ground Control Station (GCS) minimap matches the camera feed: the Woodinville-versus-SF bug.
    Runtime-*neutral* since the controller unbind: the run's builder does this for every runtime.
    """
    import nexus_sim._src.build.launch as L

    reg = Catalog.from_dict(
        {
            "vehicles": {"astro": {"usd": _usd_ref(tmp_path)}},
            "scenes": {
                "empty": {},
                "geo-scene": {
                    "usd": _scene_usd(tmp_path, "scene.usda", b'#usda 1.0\ndef Mesh "island" {}\n'),
                    "geodetic_origin": {"lat": 37.7942, "lon": -122.3954, "alt": -30.5},
                    "start": [10.0, -94.0, -8.5],
                },
            },
        }
    )
    captured = {}

    def fake_build(label, cfg, **kw):
        captured["cfg"] = cfg
        captured["kw"] = kw
        return "ORCH"

    monkeypatch.setattr(L, "build_orchestrator", fake_build)
    lc = LaunchConfig().set_vehicle("astro").set_scene("geo-scene")
    assert L.build_from_launch(lc, catalog=reg, cache_dir=tmp_path / "cache") == "ORCH"

    assert captured["cfg"]["scene_usd_path"] is not None, "the model build + render stage get the scene USD"
    assert captured["cfg"]["scene_start"] == (10.0, -94.0, -8.5), "start places the scene"
    gps = captured["cfg"]["sensors"]["gps"]["init"]
    assert (round(gps["lat"], 4), round(gps["lon"], 4)) == (37.7942, -122.3954), "GPS anchored at the scene origin"
    assert gps["alt"] == -30.5, "the scene's authored ellipsoidal alt anchors the GPS ref alt"
    assert captured["cfg"]["rtx"]["georef"] == {"lat": 37.7942, "lon": -122.3954, "alt": -30.5}, (
        "render world anchored at the same origin (alt = the surface's WGS84 ellipsoidal height, "
        "applied directly as the cesium georeference height, the retired ground-probe's replacement)"
    )


def test_build_from_launch_starts_the_px4_peer_before_the_assembly(tmp_path, monkeypatch, daemon):
    """This is the path *both* runtimes share, so the PX4 lifecycle hangs off it: the catalog's
    airframe reaches the peer, and the peer *starts*, its incremental build first, before the
    orchestrator exists: the build has to stay outside the sim's 30 s preroll window, see GH #39.
    """
    import nexus_sim._src.build.launch as L

    order = []
    monkeypatch.setattr(L, "build_orchestrator", lambda label, cfg, **kw: order.append("orchestrator") or kw)

    reg = _astro_catalog(_usd_ref(tmp_path))
    lc = LaunchConfig().set_vehicle("astro").set_scene("empty")
    kw = L.build_from_launch(lc, catalog=reg, cache_dir=tmp_path / "cache")

    started = [r["environment"].get("PX4_SIM_MODEL") for r in daemon if r.get("detach", True)]
    assert order == ["orchestrator"] and started == ["none_80001"], "the peer starts before the assembly"
    assert [p.airframe for p in kw["peers"]] == ["80001"], "the catalog's airframe reaches the peer"


# --- the controller a vehicle Universal Scene Description (USD) file declares ------------------------

PX4_ROOT = """\
def Xform "vehicle" (
    prepend apiSchemas = ["NexusPx4API"]
)
{
    string nexus:airframe = "foo"
}
"""


def _local_vehicle(tmp_path, prims: str) -> str:
    """A local vehicle file with root prim `/vehicle`, defined by `prims`."""
    path = tmp_path / "local_vehicle.usda"
    path.write_text(f'#usda 1.0\n(\n    defaultPrim = "vehicle"\n)\n\n{prims}')
    return str(path)


def _catalog(tmp_path) -> Catalog:
    """A catalog with one vehicle and no PX4 entry: the airframe lives in the vehicle's USD."""
    return Catalog.from_dict(
        {
            "vehicles": {"astro": {"usd": _usd_ref(tmp_path)}},
            "scenes": {"empty": {}},
        }
    )


def _build(tmp_path, prims: str, **kw):
    """Build a run of a local vehicle defined by `prims`."""
    import nexus_sim._src.build.launch as L

    lc = LaunchConfig().set_vehicle(_local_vehicle(tmp_path, prims)).set_scene("empty")
    return L.build_from_launch(lc, catalog=_catalog(tmp_path), cache_dir=tmp_path / "cache", **kw)


def test_the_receipt_records_the_airframe_the_vehicle_usd_declares(tmp_path):
    """The run's receipt records the airframe the vehicle USD declares.

    Given the local vehicle USD with airframe `foo`, when the run builds, then the receipt's PX4
    airframe is `foo`.
    """
    from nexus_sim._src.build.launch import resolve_vehicle_usd

    lc = LaunchConfig().set_vehicle(_local_vehicle(tmp_path, PX4_ROOT)).set_scene("empty")
    _, resolved = resolve_vehicle_usd(lc, _catalog(tmp_path), cache_dir=tmp_path / "cache")

    assert resolved.tested_config.model_dump(mode="json")["px4"] == {"airframe": "foo"}


def test_a_vehicle_that_declares_no_controller_fails_the_build(tmp_path, daemon):
    """A vehicle that declares no controller fails the build.

    Given a vehicle USD with no controller schema, when a run builds, then it raises before any peer
    starts, and the error names the vehicle's root prim.
    """
    with pytest.raises(ValueError) as err:
        _build(tmp_path, 'def Xform "vehicle"\n{\n}\n')

    assert ("/vehicle" in str(err.value), daemon) == (True, [])


class _StandInController:
    """A second controller: the loop's controller seam and nothing more."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def connect(self):
        pass

    def close(self):
        pass

    def stages(self):
        return []


def test_a_vehicle_that_declares_two_controllers_fails_the_build(tmp_path, daemon):
    """A vehicle that declares two controllers fails the build.

    Given a vehicle USD whose root prim applies two controller schemas, when a run builds, then it
    raises before any peer starts, and the error names the prim and both schemas.
    """
    from nexus_sim._src.core.registry import ComponentRegistry, default_registry

    components = ComponentRegistry(
        {"NexusPx4API": default_registry().resolve("NexusPx4API"), "StandInAPI": _StandInController}
    )
    two = PX4_ROOT.replace('["NexusPx4API"]', '["NexusPx4API", "StandInAPI"]')

    with pytest.raises(ValueError) as err:
        _build(tmp_path, two, components=components)

    named = [s in str(err.value) for s in ("/vehicle", "NexusPx4API", "StandInAPI")]
    assert (named, daemon) == ([True, True, True], [])


def test_a_controller_schema_off_the_root_prim_fails_the_build(tmp_path, daemon):
    """A controller schema off the root prim fails the build.

    Given a vehicle USD whose PX4 schema sits on a child prim, when a run builds, then it raises and
    names that prim.
    """
    child = 'def Xform "vehicle"\n{\n' + PX4_ROOT.replace('"vehicle"', '"fc"') + "}\n"

    with pytest.raises(ValueError, match="/vehicle/fc"):
        _build(tmp_path, child)


def test_a_px4_schema_with_no_airframe_authored_fails_the_build(tmp_path, daemon):
    """A PX4 schema with no airframe authored fails the build.

    Given a vehicle USD whose PX4 schema authors no airframe, when a run builds, then it raises before
    PX4 starts and names the prim, with no fallback to `astro_max`.
    """
    bare = PX4_ROOT.replace('    string nexus:airframe = "foo"\n', "")

    with pytest.raises(ValueError) as err:
        _build(tmp_path, bare)

    assert ("/vehicle" in str(err.value), daemon) == (True, [])


# --- the controller's scope: a prim of its own, of type Scope, under the root -----------------------


def _scoped(scopes: dict[str, str]) -> str:
    """A vehicle root `/vehicle`, an `Xform`, holding one `Scope` per entry of `scopes`, each applying the
    schemas its value lists, and the PX4 schema's airframe, `foo`, on the scope that applies it.
    """
    body = ""
    for name, schemas in scopes.items():
        airframe = '        string nexus:airframe = "foo"\n' if "NexusPx4API" in schemas else ""
        body += f'    def Scope "{name}" (\n        prepend apiSchemas = [{schemas}]\n    )\n    {{\n{airframe}    }}\n'
    return f'def Xform "vehicle"\n{{\n{body}}}\n'


class _Px4Peer:
    """The PX4 SITL peer in the peer mapping: it keeps the airframe the build hands it and starts no process."""

    @staticmethod
    def claim_instance():
        return 0, None

    def __init__(self, *, airframe: str, **run):
        self.airframe = airframe

    def start(self):
        pass

    def stop(self):
        pass

    def alive(self):
        return True


def _build_message(tmp_path, monkeypatch, prims: str) -> str:
    """The error a run of a local vehicle defined by `prims` fails with, or `no error`; the assembly
    step returns at once, so the vehicle needs no physics.
    """
    import nexus_sim._src.build.launch as L

    monkeypatch.setattr(L, "build_orchestrator", lambda label, cfg, **kw: kw)
    try:
        _build(tmp_path, prims)
    except ValueError as exc:
        return str(exc)
    return "no error"


@pytest.mark.parametrize("scope", ["Controller", "Autopilot"])
def test_a_vehicle_that_declares_px4_on_a_scope_flies_px4_and_starts_the_peer(tmp_path, monkeypatch, daemon, scope):
    """A vehicle that declares PX4 and the PX4 SITL peer on a `Scope` flies PX4 and starts the peer.

    Given a local vehicle whose `Scope` `/vehicle/Controller` applies `NexusPx4API` with airframe `foo`
    and `NexusPx4SitlAPI`, its root `Xform` applying neither, and a stand-in PX4 SITL peer mapped, when
    the run builds, then it flies PX4 with airframe `foo` and starts the peer once; and once with the
    scope named `/vehicle/Autopilot`.
    """
    import nexus_sim._src.build.launch as L

    monkeypatch.setattr(L, "build_orchestrator", lambda label, cfg, **kw: kw)
    vehicle = _scoped({scope: '"NexusPx4API", "NexusPx4SitlAPI"'})

    kw = _build(tmp_path, vehicle, peers={"px4_sitl": _Px4Peer})

    assert (kw["controller"].airframe, [p.airframe for p in kw["peers"]]) == ("foo", ["foo"])


@pytest.mark.parametrize(
    "prims, prim",
    [
        ('def Xform "vehicle"\n{\n' + PX4_ROOT.replace('"vehicle"', '"Controller"') + "}\n", "/vehicle/Controller"),
        (PX4_ROOT, "/vehicle"),
    ],
    ids=["xform-child", "root"],
)
def test_the_px4_schema_on_a_prim_that_is_not_a_scope_fails_the_build(tmp_path, monkeypatch, daemon, prims, prim):
    """`NexusPx4API` on a prim that isn't a `Scope` fails the build and says it applies to a `Scope`.

    Given a local vehicle whose `Xform` `/vehicle/Controller` applies `NexusPx4API` with airframe `foo`,
    when the run builds, then it fails before any peer starts, and the error names the prim and
    `NexusPx4API` and says the schema applies to a `Scope`; and once with the schema on the root `Xform`
    `/vehicle`.
    """
    message = _build_message(tmp_path, monkeypatch, prims)

    named = [f"{prim}:" in message, "NexusPx4API" in message, "Scope" in message]
    assert (named, daemon) == ([True, True, True], []), message


@pytest.mark.parametrize(
    "scopes, prim",
    [
        ({"Controller": '"NexusPx4SitlAPI"'}, "/vehicle/Controller"),
        ({"Controller": '"NexusPx4API"', "Peer": '"NexusPx4SitlAPI"'}, "/vehicle/Peer"),
    ],
    ids=["peer-alone", "peer-on-another-scope"],
)
def test_the_px4_sitl_peer_on_a_prim_with_no_px4_schema_fails_the_build(tmp_path, monkeypatch, daemon, scopes, prim):
    """The PX4 SITL peer on a prim with no `NexusPx4API` fails the build and names the prim and the peer's schema.

    Given a local vehicle whose `Scope` `/vehicle/Controller` applies `NexusPx4SitlAPI` alone, when the
    run builds, then it fails before any peer starts, and the error names `/vehicle/Controller` and
    `NexusPx4SitlAPI`; and once with `NexusPx4API` on `/vehicle/Controller` and `NexusPx4SitlAPI` on a
    second `Scope` `/vehicle/Peer`, naming `/vehicle/Peer`.
    """
    message = _build_message(tmp_path, monkeypatch, _scoped(scopes))

    named = [f"{prim}:" in message, "NexusPx4SitlAPI" in message]
    assert (named, daemon) == ([True, True], []), message


# --- the override layer a run composes over its vehicle --------------------------------------------

# A vehicle that declares PX4 and the PX4 SITL peer on its root prim, and nothing else.
PX4_SITL_ROOT = """\
def Xform "vehicle" (
    prepend apiSchemas = ["NexusPx4API", "NexusPx4SitlAPI"]
)
{
    string nexus:airframe = "foo"
}
"""


def _layer(tmp_path, body: str, name: str = "override.usda") -> Path:
    """A local override layer whose prims are `body`."""
    path = tmp_path / name
    path.write_text(f"#usda 1.0\n\n{body}")
    return path


def _layered(tmp_path, vehicle: str, layer: Path | str) -> LaunchConfig:
    """A launch of `vehicle`, a catalog name or a local path, in the empty scene, with the override layer `layer`."""
    return LaunchConfig.from_dict({"vehicle": vehicle, "scene": "empty", "layer": str(layer)})


class _Handed(Px4Fake):
    """The fake PX4 in the peer mapping, which keeps the airframe the build hands it."""

    def __init__(self, *, airframe: str, **run):
        super().__init__(**run)
        self.airframe = airframe


def _referenced(tmp_path, metadata: str = "", contents: str = "") -> str:
    """A local vehicle whose root prim `/vehicle` references the local fixture vehicle of
    ``tests/usd/sensor_vehicle.py``, which flies airframe `astro_max`, and declares the PX4 SITL peer,
    with `metadata` and `contents` of the root prim's own.
    """
    path = tmp_path / "referencing_vehicle.usda"
    path.write_text(
        f'#usda 1.0\n(\n    defaultPrim = "vehicle"\n)\n\n'
        f'def Xform "vehicle" (\n    references = @{sv.BASE}@\n'
        f'    prepend apiSchemas = ["NexusPx4SitlAPI"]\n{metadata})\n{{\n{contents}}}\n'
    )
    return str(path)


def _handed_airframe(tmp_path, vehicle: str, layer: Path) -> list[str]:
    """Build `vehicle` over `layer` for real, the PX4 SITL peer sent to a fake that keeps its airframe,
    and return the airframe the build handed each PX4 peer of the run.
    """
    import nexus_sim._src.build.launch as L

    launch = LaunchConfig.from_dict(
        {"vehicle": vehicle, "scene": sv.SCENE, "layer": str(layer), "runtime": {"device": "cpu"}}
    )
    loop = L.build_from_launch(launch, preroll_timeout=10.0, peers={"px4_sitl": _Handed})
    handed = [peer.airframe for peer in loop.peers]
    loop.close()
    return handed


def test_a_layer_that_changes_a_declared_value_changes_what_the_run_builds(tmp_path, warp_cpu):
    """A layer that changes a declared value changes what the run builds.

    Given a vehicle that references the local fixture vehicle, whose PX4 schema declares airframe
    `astro_max`, and a layer that sets `nexus:airframe` to `bar` on its root prim, when the run builds
    with the PX4 SITL peer sent to a fake, then the build hands the fake airframe `bar`.
    """
    layer = _layer(tmp_path, 'over "vehicle"\n{\n    string nexus:airframe = "bar"\n}\n')

    assert _handed_airframe(tmp_path, _referenced(tmp_path), layer) == ["bar"]


def test_a_layer_that_selects_a_variant_builds_that_variant(tmp_path, warp_cpu):
    """A layer that selects a variant builds that variant.

    Given a vehicle that references the local fixture vehicle and adds an `airframe` variant set whose
    selection is `a`, and a layer that selects its second variant, `b`, when the run builds with the
    PX4 SITL peer sent to a fake, then the build hands the fake variant `b`'s airframe, `b`.
    """
    metadata = '    variants = {\n        string airframe = "a"\n    }\n    prepend variantSets = "airframe"\n'
    contents = (
        '    variantSet "airframe" = {\n        "a" {\n            string nexus:airframe = "a"\n        }\n'
        '        "b" {\n            string nexus:airframe = "b"\n        }\n    }\n'
    )
    layer = _layer(tmp_path, 'over "vehicle" (\n    variants = {\n        string airframe = "b"\n    }\n)\n{\n}\n')

    assert _handed_airframe(tmp_path, _referenced(tmp_path, metadata, contents), layer) == ["b"]


def test_a_layer_that_deactivates_a_declaration_builds_nothing_for_it(tmp_path, warp_cpu):
    """A layer that deactivates a declaration builds nothing for it.

    Given the local fixture vehicle with an Inertial Measurement Unit (IMU), a magnetometer, a barometer
    and a Global Positioning System (GPS) receiver, and a layer that deactivates the magnetometer's
    prim, when the run builds with the PX4 SITL peer sent to its fake, then it builds the IMU, the
    barometer and the GPS receiver, and no magnetometer.
    """
    import nexus_sim._src.build.launch as L

    sensors = "".join(
        sv.prim(name, schema)
        for name, schema in (
            ("Imu", "NexusImuAPI"),
            ("Mag", "NexusMagAPI"),
            ("Baro", "NexusBaroAPI"),
            ("Gps", "NexusGpsAPI"),
        )
    )
    vehicle = sv.vehicle(tmp_path, sensors, px4=True)
    layer = _layer(
        tmp_path,
        'over "vehicle"\n{\n    over "body"\n    {\n'
        '        over "Mag" (\n            active = false\n        )\n        {\n        }\n'
        "    }\n}\n",
    )
    launch = LaunchConfig.from_dict(
        {"vehicle": vehicle, "scene": sv.SCENE, "layer": str(layer), "runtime": {"device": "cpu"}}
    )

    loop = L.build_from_launch(launch, preroll_timeout=10.0, peers={"px4_sitl": Px4Fake})
    built = [type(s).__name__ for s in loop.sensors]
    loop.close()

    assert built == ["ImuSensor", "BaroSensor", "GpsSensor"]


def test_the_receipt_records_the_layers_hash_beside_the_vehicle_assets(tmp_path):
    """The receipt records the layer's hash beside the vehicle asset's.

    Given a run with a layer, when it resolves, then its receipt holds the layer's sha256; and after a
    one-byte change to the layer, the receipt holds the new sha256 instead.
    """
    from nexus_sim._src.config import resolve

    vehicle = _local_vehicle(tmp_path, PX4_SITL_ROOT)
    layer = _layer(tmp_path, 'over "vehicle"\n{\n    string nexus:airframe = "bar"\n}\n')
    before = hashlib.sha256(layer.read_bytes()).hexdigest()
    first = resolve(_layered(tmp_path, vehicle, layer), _catalog(tmp_path), cache_dir=tmp_path / "cache")
    layer.write_text(layer.read_text().replace('"bar"', '"baz"'))
    after = hashlib.sha256(layer.read_bytes()).hexdigest()
    second = resolve(_layered(tmp_path, vehicle, layer), _catalog(tmp_path), cache_dir=tmp_path / "cache")

    first_json, second_json = first.tested_config.to_json(), second.tested_config.to_json()
    assert (before in first_json, after in second_json, before in second_json) == (True, True, False)


def test_a_run_with_no_layer_builds_and_records_as_today(tmp_path, monkeypatch, warp_cpu):
    """A run with no layer builds and records as today.

    Given the catalog vehicle `astro_max_base` and no layer, when the run builds with the PX4 SITL peer
    sent to its fake, then it builds the PX4 controller on airframe `astro_max`, its IMU, magnetometer,
    barometer and Global Positioning System (GPS) sensors, and one PX4 peer; and its receipt names the
    vehicle, the airframe and the scene, and no layer or geodetic origin.
    """
    import nexus_sim._src.build.launch as L

    monkeypatch.delenv("NEXUS_ASSET_CACHE", raising=False)  # the shipped vehicle comes from the checkout's own cache
    monkeypatch.chdir(tmp_path)  # no project catalog: only the bundled one
    launch = LaunchConfig.from_dict({"vehicle": "astro_max_base", "scene": "empty", "runtime": {"device": "cpu"}})

    loop = L.build_from_launch(launch, preroll_timeout=10.0, peers={"px4_sitl": Px4Fake})
    built = (
        type(loop.controller).__name__,
        loop.controller.airframe,
        [type(s).__name__ for s in loop.sensors],
        [type(p).__name__ for p in loop.peers],
    )
    loop.close()
    receipt = L.resolve_vehicle_usd(launch)[1].tested_config.model_dump(mode="json")
    recorded = {k: receipt[k] for k in ("vehicle", "layer", "px4", "scene", "scene_start", "geodetic_origin")}

    assert (built, recorded) == (
        ("Px4MavlinkController", "astro_max", ["ImuSensor", "MagSensor", "BaroSensor", "GpsSensor"], ["Px4Fake"]),
        {
            "vehicle": "astro_max_base",
            "layer": None,
            "px4": {"airframe": "astro_max"},
            "scene": "empty",
            "scene_start": None,
            "geodetic_origin": None,
        },
    )


def _px4_run(tmp_path, device: str):
    """Build the local fixture vehicle, which declares PX4 and its SITL peer, in the fixture scene on the PX4 fake,
    and return the loop and the receipt it carries, the one the build handed it, as JSON.
    """
    import nexus_sim._src.build.launch as L

    launch = LaunchConfig.from_dict(
        {"vehicle": sv.vehicle(tmp_path, px4=True), "scene": sv.SCENE, "runtime": {"device": device}}
    )
    loop = L.build_from_launch(launch, preroll_timeout=10.0, peers={"px4_sitl": Px4Fake})
    return loop, loop.settings


def test_a_px4_runs_receipt_carries_no_substeps_determinism_or_sensors(tmp_path, warp_cpu):
    """A PX4 run's receipt carries no `substeps`, no `determinism` and no `sensors`.

    Given a launch of the fixture vehicle in the fixture scene on the PX4 fake, when the run builds and a caller reads
    its receipt as JSON, then `runtime` holds exactly `device`, `seed`, `dt`, `max_steps`, `rtf` and
    `solver`, and the receipt has no `sensors` key.
    """
    loop, receipt = _px4_run(tmp_path, "cpu")
    loop.close()

    assert (sorted(receipt["runtime"]), "sensors" in receipt) == (
        ["device", "dt", "max_steps", "rtf", "seed", "solver"],
        False,
    )


def test_the_receipt_of_a_run_on_auto_records_the_device_the_run_picked(tmp_path, warp_cpu):
    """The receipt of a run on `auto` records the device the run picked, never `auto`.

    Given a launch on `runtime.device: auto` on the PX4 fake, when the run builds, then its receipt's
    `runtime.device` is `cuda` on a box with CUDA and `cpu` on a box without, the same kind as the Warp
    device the loop runs on.
    """
    import warp as wp

    loop, receipt = _px4_run(tmp_path, "auto")
    ran_on = "cuda" if loop.physics.model.device.is_cuda else "cpu"
    loop.close()

    box = "cuda" if wp.is_cuda_available() else "cpu"
    assert (receipt["runtime"]["device"], ran_on) == (box, box)


def test_a_run_on_an_explicit_cpu_runs_on_the_cpu_and_records_cpu(tmp_path, warp_cpu):
    """A run on an explicit `cpu` still runs on the CPU and records `cpu`.

    Given a launch on `runtime.device: cpu` on the PX4 fake, on a box with or without CUDA, when the run
    builds, then the loop's Warp device is `cpu` and the receipt's `runtime.device` is `cpu`.
    """
    loop, receipt = _px4_run(tmp_path, "cpu")
    ran_on = str(loop.physics.model.device)
    loop.close()

    assert (ran_on, receipt["runtime"]["device"]) == ("cpu", "cpu")


def test_a_px4_run_steps_the_physics_once_per_control_tick(tmp_path, warp_cpu, caplog):
    """A PX4 run still steps the physics once per control tick.

    Given a launch of the fixture vehicle in the fixture scene on the PX4 fake, when the run builds and ticks five
    times, then every tick runs, and the stage plan the run logs at its first tick, the ring every tick
    runs, names the physics `step` stage once, so the five ticks step the physics five times.
    """
    loop, _ = _px4_run(tmp_path, "cpu")
    with caplog.at_level(logging.INFO, logger="nexus"):
        ticked = [loop.step() for _ in range(5)]
    loop.close()

    plans = [r.getMessage() for r in caplog.records if r.getMessage().startswith("stage plan:")]
    steps_per_tick = [len(re.findall(r"\bstep\b", plan)) for plan in plans]
    assert (ticked, steps_per_tick) == ([True] * 5, [1])


def test_a_px4_run_takes_no_estimator_and_its_stages_stay_the_same(tmp_path, warp_cpu, caplog):
    """A PX4 run takes no estimator, and its stages stay the same.

    Given the fixture vehicle on the PX4 fake, when the run takes its first tick, then its logged stage plan
    equals today's, with no estimator stage.
    """
    loop, _ = _px4_run(tmp_path, "cpu")
    with caplog.at_level(logging.INFO, logger="nexus"):
        loop.step()
    loop.close()

    plans = [r.getMessage() for r in caplog.records if r.getMessage().startswith("stage plan:")]
    assert plans == [
        "stage plan: eager(clear -> rotors -> propellers -> step -> record) host(read) host(truth) host(exchange)"
    ]


def test_a_layer_path_that_does_not_exist_fails_before_any_peer_starts(tmp_path, daemon):
    """A layer path that doesn't exist fails before any peer starts.

    Given a layer path with no file behind it, when the run builds, then it raises
    `FileNotFoundError` naming the path, and no peer has started.
    """
    import nexus_sim._src.build.launch as L

    missing = tmp_path / "no_such_layer.usda"
    launch = _layered(tmp_path, _local_vehicle(tmp_path, PX4_SITL_ROOT), missing)

    with pytest.raises(FileNotFoundError) as err:
        L.build_from_launch(launch, catalog=_catalog(tmp_path), cache_dir=tmp_path / "cache")

    assert (str(missing) in str(err.value), daemon) == (True, [])


def test_a_px4_sitl_peer_declared_without_the_px4_controller_fails_the_build(tmp_path, daemon):
    """A PX4 SITL peer declared without the PX4 controller fails the build.

    Given a fixture vehicle whose root prim declares the PX4 SITL peer and no `NexusPx4API`, when the
    run builds, then it raises naming the prim and the peer's schema, and no peer has started.
    """
    peer_only = 'def Xform "vehicle" (\n    prepend apiSchemas = ["NexusPx4SitlAPI"]\n)\n{\n}\n'

    with pytest.raises(ValueError) as err:
        _build(tmp_path, peer_only)

    named = [s in str(err.value) for s in ("/vehicle", "NexusPx4SitlAPI")]
    assert (named, daemon) == ([True, True], [])
