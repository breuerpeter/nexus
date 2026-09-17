"""``nexus`` command-line tool parsing/validation: the command surface only, no runtime boots.

The test monkeypatches the dispatch targets: ``_run_standalone``, the Sim path; ``resolve_runtime``, the
auto selection driven by the vehicle's Universal Scene Description (USD); and ``run_isaacsim``. No Kit, no sim.
"""

import importlib

import pytest

cli = importlib.import_module("nexus._src.cli.main")
detect = importlib.import_module("nexus._src.cli.detect")


def _run(monkeypatch, argv, runtime="standalone"):
    """Invoke the command-line tool with *argv*; a stub pins auto-resolution to *runtime*."""
    captured = {}

    def fake_standalone(args):
        captured.update(runtime="standalone", args=args)

    monkeypatch.setattr(cli, "_run_standalone", fake_standalone)
    monkeypatch.setattr(detect, "resolve_runtime", lambda args: runtime if args.runtime == "auto" else args.runtime)
    monkeypatch.setattr("sys.argv", ["nexus", *argv])
    cli.main()
    return captured


def test_auto_default_routes_standalone_without_rtx(monkeypatch):
    """--runtime defaults to auto; a vehicle without RTX prims routes to the standalone Sim path."""
    cap = _run(monkeypatch, ["run"])
    assert cap["runtime"] == "standalone"
    assert cap["args"].runtime == "auto"


def test_stream_on_a_cameraless_vehicle_errors(monkeypatch):
    """--stream needs RTX camera sensors; a vehicle that resolves standalone rejects it."""
    with pytest.raises(SystemExit):
        _run(monkeypatch, ["run", "--stream"], runtime="standalone")


def test_explicit_standalone_skips_detection(monkeypatch):
    cap = _run(monkeypatch, ["run", "--runtime", "standalone"])
    assert cap["runtime"] == "standalone"


def test_log_and_view_are_exclusive(monkeypatch):
    with pytest.raises(SystemExit):
        _run(monkeypatch, ["run", "--log", "--view"])


def test_rtx_prim_scan_reads_the_authored_camera(tmp_path):
    """A vehicle USD carrying a camera prim → the auto-runtime detector finds it.

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
