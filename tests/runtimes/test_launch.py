"""The launch glue: resolve a vehicle Universal Scene Description (USD) via newton-config + route by
control kind.

Exercises the resolution + builder-construction + routing seam, not the heavy NewtonPhysics build,
which needs a real newton-loadable USD asset. The resolved USD is a file:// asset, sha-verified by
the real newton-assets resolver.
"""

import hashlib
from pathlib import Path

import pytest

from nexus._src.config import LaunchConfig, Registry


@pytest.fixture(autouse=True)
def daemon(monkeypatch, tmp_path):
    """``build_from_launch`` starts the PX4 Software In The Loop (SITL) peer, which builds and runs PX4
    in docker. These tests exercise the resolution + routing seam, so the docker daemon is a stand-in
    that records what the build asks of it, and ``$PX4_DIR`` a stand-in tree, so nothing fetches.
    """
    from docker.errors import NotFound

    import nexus._src.peers.containers as containers

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


def _registry(usd_ref: dict) -> Registry:
    return Registry.from_dict(
        {
            "vehicles": [{"name": "astro", "usd": usd_ref, "px4": {"airframe": "80001"}}],
            "scenes": {"empty": {}},
            "defaults": {"vehicle": "astro", "scene": "empty"},
        }
    )


def _usd_ref(tmp_path) -> dict:
    blob = tmp_path / "vehicle.usda"
    blob.write_bytes(b"#usda 1.0\n")
    return {"url": blob.as_uri(), "sha256": hashlib.sha256(blob.read_bytes()).hexdigest(), "filename": "vehicle.usda"}


def test_resolve_to_vehicle_builder_uses_resolved_usd(tmp_path):
    from nexus._src.build.launch import resolve_to_vehicle_builder

    ref = _usd_ref(tmp_path)
    reg = _registry(ref)
    builder, resolved = resolve_to_vehicle_builder(
        LaunchConfig().set_vehicle("astro"), reg, cache_dir=tmp_path / "cache"
    )
    # the builder points at the verified, content-addressed local copy of the USD
    assert Path(builder.cfg["usd_path"]).read_bytes() == b"#usda 1.0\n"
    assert ref["sha256"] in str(builder.cfg["usd_path"])  # content-addressed cache path
    assert resolved.tested_config.px4.airframe == "80001"


def test_non_px4_control_kinds_are_rejected():
    """PX4 is the one first-class control kind: everything else is an example that self-assembles
    via Sim.from_orchestrator; the Control schema rejects the old kinds at validation time.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LaunchConfig().set_control("builtin")
    with pytest.raises(ValidationError):
        LaunchConfig().set_control("policy")


def test_scenario_from_launch_honors_dt_seed_device(tmp_path):
    from nexus._src.build.launch import _scenario_from_launch

    lc = LaunchConfig.from_dict({"vehicle": "astro", "runtime": {"dt": 0.01, "seed": 7, "device": "cpu"}})
    cfg = _scenario_from_launch(lc)
    assert cfg["physics"]["dt"] == 0.01
    assert cfg["physics"]["force_cpu"] is True
    assert cfg["seed"] == 7

    cfg_gpu = _scenario_from_launch(LaunchConfig.from_dict({"runtime": {"device": "cuda:0"}}))
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
    Runtime-*neutral* since the controller unbind: the one launch glue does this for every runtime.
    """
    import nexus._src.build.launch as L

    reg = Registry.from_dict(
        {
            "vehicles": [{"name": "astro", "usd": _usd_ref(tmp_path), "px4": {"airframe": "80001"}}],
            "scenes": {
                "empty": {},
                "geo-scene": {
                    "usd": _scene_usd(tmp_path, "scene.usda", b'#usda 1.0\ndef Mesh "island" {}\n'),
                    "geodetic_origin": {"lat": 37.7942, "lon": -122.3954, "alt": -30.5},
                    "start": [10.0, -94.0, -8.5],
                },
            },
            "defaults": {"vehicle": "astro", "scene": "empty"},
        }
    )
    captured = {}

    def fake_build(label, cfg, **kw):
        captured["cfg"] = cfg
        captured["kw"] = kw
        return "ORCH"

    monkeypatch.setattr(L, "build_orchestrator", fake_build)
    lc = LaunchConfig().set_vehicle("astro").set_control("px4-sitl").set_scene("geo-scene")
    assert L.build_from_launch(lc, registry=reg, cache_dir=tmp_path / "cache") == "ORCH"

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
    """This is the seam *both* runtimes share, so the PX4 lifecycle hangs off it: the registry's
    airframe reaches the peer, and the peer *starts*, its incremental build first, before the
    orchestrator exists: the build has to stay outside the sim's 30 s preroll window, see GH #39.
    """
    import nexus._src.build.launch as L

    order = []
    monkeypatch.setattr(L, "build_orchestrator", lambda label, cfg, **kw: order.append("orchestrator") or kw)

    reg = _registry(_usd_ref(tmp_path))
    lc = LaunchConfig().set_vehicle("astro").set_control("px4-sitl")
    kw = L.build_from_launch(lc, registry=reg, cache_dir=tmp_path / "cache")

    started = [r["environment"].get("PX4_SIM_MODEL") for r in daemon if r.get("detach", True)]
    assert order == ["orchestrator"] and started == ["none_80001"], "the peer starts before the assembly"
    assert [p.airframe for p in kw["peers"]] == ["80001"], "the registry airframe reaches the peer"
