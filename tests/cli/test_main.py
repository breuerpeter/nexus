"""``nexus`` command-line tool parsing/validation: the command surface only, no run starts."""

import hashlib
import importlib

import pytest

cli = importlib.import_module("nexus._src.cli.main")

PLAIN_USD = '#usda 1.0\n(\n    defaultPrim = "vehicle"\n)\ndef Xform "vehicle" {}\n'


def test_log_and_view_are_exclusive(monkeypatch):
    monkeypatch.setattr("sys.argv", ["nexus", "run", "--log", "--view"])
    with pytest.raises(SystemExit):
        cli.main()


def test_stream_on_a_vehicle_without_a_camera_fails_before_the_run(monkeypatch, tmp_path):
    """--stream publishes the camera feeds, so a vehicle that authors no camera has nothing to stream."""
    pytest.importorskip("pxr")
    usd = tmp_path / "plain.usda"
    usd.write_text(PLAIN_USD)
    registry = tmp_path / "catalog.yaml"
    registry.write_text(
        "vehicles:\n  - name: plain\n"
        f'    usd: {{ url: "{usd.as_uri()}", sha256: {hashlib.sha256(usd.read_bytes()).hexdigest()} }}\n'
        "scenes:\n  empty: {}\n"
    )
    monkeypatch.setenv("NEXUS_ASSET_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr("sys.argv", ["nexus", "run", "--stream", "--vehicle", "plain", "--registry", str(registry)])
    with pytest.raises(ValueError, match="camera"):
        cli.main()


def test_rtx_prim_scan_reads_the_authored_camera(tmp_path):
    """A vehicle Universal Scene Description (USD) file with a camera prim under its root prim: the scan that starts Kit finds it.

    Deterministic and self-contained: author a synthetic vehicle USD with an FpvCam camera prim
    via pxr, then scan it: no cached/downloaded asset, no network, no prep scripts.
    """
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage, "/Vehicle")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    stage.SetDefaultPrim(body.GetPrim())
    UsdGeom.Camera.Define(stage, "/Vehicle/FpvCam")
    out = tmp_path / "vehicle.usda"
    stage.GetRootLayer().Export(str(out))

    from nexus._src.vehicle.sensors.usd import vehicle_rtx_sensor_prims

    prims = vehicle_rtx_sensor_prims(str(out))
    assert any("FpvCam" in p for p in prims)
