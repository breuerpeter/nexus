"""``nexus`` command-line tool parsing/validation: the command surface only, no run starts."""

import hashlib
import importlib

import pytest

cli = importlib.import_module("nexus._src.cli.main")

# A vehicle that declares PX4 and authors no camera.
PLAIN_USD = (
    '#usda 1.0\n(\n    defaultPrim = "vehicle"\n)\n'
    'def Xform "vehicle" (\n    prepend apiSchemas = ["NexusPx4API"]\n)\n{\n    string nexus:airframe = "astro_max"\n}\n'
)


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


def test_script_is_no_subcommand(monkeypatch, tmp_path):
    """`script` is no subcommand: `nexus` run with `script scripts/assets/obj_to_usd.py` exits with a usage error.

    The Kit-only asset scripts start Kit themselves, so the tool carries no way to run a file in the
    Kit image. `DOCKER_HOST` points nowhere so that no container can start if the tool still tries.
    """
    monkeypatch.setenv("DOCKER_HOST", f"unix://{tmp_path / 'no-daemon.sock'}")
    monkeypatch.setattr("sys.argv", ["nexus", "script", "scripts/assets/obj_to_usd.py"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 2  # argparse's usage error


def test_help_names_no_script_subcommand(monkeypatch, capsys):
    """`script` is no subcommand: `nexus --help` names no `script`."""
    monkeypatch.setattr("sys.argv", ["nexus", "--help"])
    with pytest.raises(SystemExit):
        cli.main()
    assert "script" not in capsys.readouterr().out


def test_control_is_no_flag_of_the_command_line_tool(monkeypatch, tmp_path, capsys):
    """The command-line tool takes no `--control` flag.

    Given `nexus run --control px4-sitl`, when it parses, then it exits non-zero with argparse's
    unrecognized-argument error. The vehicle names a file that doesn't exist and `DOCKER_HOST` points
    nowhere, so no run can start if the tool still takes the flag.
    """
    monkeypatch.setenv("DOCKER_HOST", f"unix://{tmp_path / 'no-daemon.sock'}")
    missing = tmp_path / "missing.usda"
    monkeypatch.setattr("sys.argv", ["nexus", "run", "--control", "px4-sitl", "--vehicle", str(missing)])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert (e.value.code, "unrecognized arguments: --control" in capsys.readouterr().err) == (2, True)


def _said_when_run(monkeypatch, capsys, tmp_path, *argv):
    """Whether `nexus run *argv` fails, and what it says when it stops, offline and with no Docker daemon.

    An empty asset cache and a dead proxy keep the run from fetching an asset, and `DOCKER_HOST`
    points nowhere, so no peer can start if the tool still launches the run.
    """
    monkeypatch.setenv("NEXUS_ASSET_CACHE", str(tmp_path / "cache"))
    for proxy in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
        monkeypatch.setenv(proxy, "http://127.0.0.1:9")  # the discard port: no proxy answers
    monkeypatch.setenv("DOCKER_HOST", f"unix://{tmp_path / 'no-daemon.sock'}")
    monkeypatch.setattr("sys.argv", ["nexus", "run", *argv])
    with pytest.raises(BaseException) as e:
        cli.main()
    code = e.value.code if isinstance(e.value, SystemExit) else 1  # an uncaught exception exits 1
    return code not in (0, None), str(e.value) + capsys.readouterr().err


def test_a_run_that_names_no_vehicle_fails_before_it_starts(monkeypatch, capsys, tmp_path):
    """A command-line run that names no vehicle fails before it starts, and the error says to name one.

    Given the bundled catalog, when `nexus run --scene empty` names no `--vehicle`, then it exits
    non-zero with an error that names `--vehicle`, and no peer starts.
    """
    failed, said = _said_when_run(monkeypatch, capsys, tmp_path, "--scene", "empty")
    assert (failed, "--vehicle" in said) == (True, True)


def test_a_run_that_names_no_scene_fails_before_it_starts(monkeypatch, capsys, tmp_path):
    """A command-line run that names no scene fails before it starts, and the error says to name one.

    Given the bundled catalog, when `nexus run --vehicle astro_max_base` names no `--scene`, then it
    exits non-zero with an error that names `--scene`, and no peer starts.
    """
    failed, said = _said_when_run(monkeypatch, capsys, tmp_path, "--vehicle", "astro_max_base")
    assert (failed, "--scene" in said) == (True, True)


def test_a_run_on_a_catalog_with_a_defaults_block_fails_and_says_to_name_the_vehicle_and_scene(
    monkeypatch, capsys, tmp_path
):
    """A project catalog with a `defaults` block fails to load, and the error names the removal and
    says to name the vehicle and scene on the run.

    Given a `nexus.registry.yaml` with `defaults: { vehicle: my_quad }`, when a run that names its
    vehicle and scene loads it, then it fails with an error that names `defaults` as removed and
    points at `--vehicle` and `--scene`.
    """
    catalog = tmp_path / "nexus.registry.yaml"
    catalog.write_text(
        'vehicles:\n  - name: my_quad\n    usd: { url: "file:///my_quad.usda", sha256: abc }\n'
        "defaults: { vehicle: my_quad }\n"
    )
    argv = ("--registry", str(catalog), "--vehicle", "astro_max_base", "--scene", "empty")
    failed, said = _said_when_run(monkeypatch, capsys, tmp_path, *argv)
    assert (failed, [w in said for w in ("defaults", "remov", "--vehicle", "--scene")]) == (True, [True] * 4)
