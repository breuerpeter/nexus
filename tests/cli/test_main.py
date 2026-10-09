"""``nexus`` command-line tool parsing/validation: the command surface only, no run starts."""

import importlib

import pytest

cli = importlib.import_module("nexus_sim._src.cli.main")

# A vehicle that declares PX4 on its controller's scope and authors no camera.
PLAIN_USD = (
    '#usda 1.0\n(\n    defaultPrim = "vehicle"\n)\n'
    'def Xform "vehicle"\n{\n    def Scope "Controller" (\n        prepend apiSchemas = ["NexusPx4API"]\n    )\n'
    '    {\n        string nexus:airframe = "astro_max"\n    }\n}\n'
)


def test_log_and_view_are_exclusive(monkeypatch):
    monkeypatch.setattr("sys.argv", ["nexus", "run", "--log", "--view"])
    with pytest.raises(SystemExit):
        cli.main()


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
    monkeypatch.setattr(
        "sys.argv", ["nexus", "run", "--control", "px4-sitl", "--vehicle", str(missing), "--scene", "empty"]
    )
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert (e.value.code, "unrecognized arguments: --control" in capsys.readouterr().err) == (2, True)


def test_nexus_run_rejects_stream(monkeypatch, tmp_path, capsys):
    """The `nexus run` command rejects `--stream`: a camera's feed has nowhere to publish until a vehicle
    declares a companion.

    Given `nexus run --stream`, when it parses, then it exits with argparse's unrecognized-argument
    error. The vehicle names a file that doesn't exist and `DOCKER_HOST` points nowhere, so no run can
    start if the tool still takes the flag.
    """
    monkeypatch.setenv("DOCKER_HOST", f"unix://{tmp_path / 'no-daemon.sock'}")
    missing = tmp_path / "missing.usda"
    monkeypatch.setattr("sys.argv", ["nexus", "run", "--stream", "--vehicle", str(missing), "--scene", "empty"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert (e.value.code, "unrecognized arguments: --stream" in capsys.readouterr().err) == (2, True)


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

    Given a `nexus.catalog.yaml` with `defaults: { vehicle: my_quad }`, when a run that names its
    vehicle and scene loads it, then it fails with an error that names `defaults` as removed and
    points at `--vehicle` and `--scene`.
    """
    catalog = tmp_path / "nexus.catalog.yaml"
    catalog.write_text(
        'vehicles:\n  my_quad:\n    usd: { url: "file:///my_quad.usda", sha256: abc }\ndefaults: { vehicle: my_quad }\n'
    )
    argv = ("--catalog", str(catalog), "--vehicle", "astro_max_base", "--scene", "empty")
    failed, said = _said_when_run(monkeypatch, capsys, tmp_path, *argv)
    assert (failed, [w in said for w in ("defaults", "remov", "--vehicle", "--scene")]) == (True, [True] * 4)


def test_a_run_told_its_catalog_flies_the_vehicles_of_that_file(monkeypatch, capsys, tmp_path):
    """`nexus run --catalog <file>` and `Sim(..., catalog=<file>)` fly the vehicles of that file
    beside the bundled ones.

    Given a catalog file outside the working directory that lists a vehicle at a local Universal Scene
    Description (USD) path, when the command line parses `--catalog <file>`, and when `Sim` takes
    `catalog=<file>`, then the vehicle resolves from that file in both: each run goes on to fetch the
    vehicle's file that the catalog lists, which stops it, since the entry's hash is a stand-in.
    """
    from nexus_sim import Sim

    usd = tmp_path / "assets" / "project_vehicle.usda"
    usd.parent.mkdir()
    usd.write_text(PLAIN_USD)
    catalog = tmp_path / "elsewhere" / "project.yaml"
    catalog.parent.mkdir()
    catalog.write_text(f'vehicles:\n  project_vehicle:\n    usd: {{ url: "file://{usd}", sha256: "0" }}\n')
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    argv = ("--catalog", str(catalog), "--vehicle", "project_vehicle", "--scene", "empty")
    _, by_cli = _said_when_run(monkeypatch, capsys, tmp_path, *argv)
    with pytest.raises(BaseException) as by_sim:
        with Sim("project_vehicle", scene="empty", catalog=str(catalog), device="cpu"):
            pass

    assert (usd.name in by_cli, usd.name in str(by_sim.value)) == (True, True)


# --- the override layer, and the PX4 flags that go -------------------------------------------------


class _Daemon:
    """A stand-in docker daemon that records each container run and holds no image to build."""

    def __init__(self):
        from docker.errors import NotFound

        self.runs: list[dict] = []
        daemon = self

        class _Container:
            def remove(self, force=False):
                pass

        class Images:
            def get(self, tag):
                return object()

        class Containers:
            def run(self, image, **kwargs):
                daemon.runs.append({"image": image, **kwargs})
                return b"" if not kwargs.get("detach", True) else _Container()

            def get(self, name):
                raise NotFound(name)

            def list(self, **kwargs):
                return []

        self.images = Images()
        self.containers = Containers()


def test_the_command_line_takes_the_layer(monkeypatch, tmp_path, warp_cpu):
    """The command line takes the layer.

    Given a local vehicle whose PX4 schema declares airframe `foo`, a layer that sets it to `bar`, and a
    stand-in docker daemon, when `nexus run --vehicle … --scene empty --layer <layer>` runs, then PX4 Software
    In The Loop (SITL) starts on the layer's airframe, `PX4_SIM_MODEL=none_bar`, as the same run through `Sim` does.
    """
    import nexus_sim._src.peers.containers as containers
    from nexus_sim._src.api.sim import Sim

    daemon = _Daemon()
    monkeypatch.setattr(containers, "client", lambda: daemon)
    monkeypatch.setattr(containers, "_pump_logs", lambda container, log_path: None)
    px4 = tmp_path / "px4"
    px4.mkdir()
    (px4 / "Makefile").write_text("px4_sitl:\n")  # what marks a folder as a PX4 tree
    monkeypatch.setenv("PX4_DIR", str(px4))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("NEXUS_ASSET_CACHE", str(tmp_path / "cache"))
    vehicle = tmp_path / "vehicle.usda"
    vehicle.write_text(
        PLAIN_USD.replace('["NexusPx4API"]', '["NexusPx4API", "NexusPx4SitlAPI"]').replace('"astro_max"', '"foo"')
    )
    layer = tmp_path / "override.usda"
    layer.write_text(
        '#usda 1.0\n\nover "vehicle"\n{\n    over "Controller"\n    {\n        string nexus:airframe = "bar"\n    }\n}\n'
    )

    def started(run) -> list[str]:
        daemon.runs.clear()
        try:
            run()
        except BaseException:  # the bare vehicle has no bodies, so the run stops after its peer starts
            pass
        return [r["environment"].get("PX4_SIM_MODEL") for r in daemon.runs if r.get("detach", True)]

    argv = ["nexus", "run", "--vehicle", str(vehicle), "--scene", "empty", "--device", "cpu", "--layer", str(layer)]
    monkeypatch.setattr("sys.argv", argv)
    by_cli = started(cli.main)

    def through_sim():
        with Sim(str(vehicle), scene="empty", device="cpu", layer=str(layer)):
            pass

    by_sim = started(through_sim)

    assert (by_cli, by_sim) == (["none_bar"], ["none_bar"])


def test_the_px4_flags_are_gone(monkeypatch, tmp_path, capsys):
    """Neither the command line nor `Sim` takes a PX4 flag.

    Given `nexus run` with `--px4 external` or `--px4-instance 1`, when it parses, then it exits with
    argparse's unrecognized-argument error; and `Sim(px4=…)` or `Sim(px4_instance=…)` raises
    `TypeError`. The vehicle names a file that doesn't exist and `DOCKER_HOST` points nowhere, so no
    run can start if the tool still takes a flag.
    """
    from nexus_sim._src.api.sim import Sim

    monkeypatch.setenv("DOCKER_HOST", f"unix://{tmp_path / 'no-daemon.sock'}")
    missing = str(tmp_path / "missing.usda")

    def rejected(*flag) -> bool:
        monkeypatch.setattr("sys.argv", ["nexus", "run", *flag, "--vehicle", missing, "--scene", "empty"])
        try:
            cli.main()
            return False
        except SystemExit as e:
            return e.code == 2 and f"unrecognized arguments: {flag[0]}" in capsys.readouterr().err
        except BaseException:
            return False

    def refused(**kwargs) -> bool:
        try:
            Sim("astro_max_base", scene="empty", **kwargs)
            return False
        except TypeError:
            return True

    flags = (rejected("--px4", "external"), rejected("--px4-instance", "1"))
    kwargs = (refused(px4="external"), refused(px4_instance=1))
    assert (flags, kwargs) == ((True, True), (True, True))
