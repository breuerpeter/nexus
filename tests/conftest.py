"""Shared test helpers."""

import re
import subprocess
import sys

import newton_usd_schemas  # noqa: F401
import pytest
import warp as wp

# The schema plugins the tests apply, registered here, before any test module opens a stage: OpenUSD
# builds its schema registry once, on first use, and a plugin registered after that never shows in it.
# Newton's plugin backs the test that applies a Newton schema, and the stand-in project's backs the tests
# that apply `StandInAPI`.
import tests.usd.stand_in.nexus_stand_in  # noqa: F401

# Warp's default device when the session starts. A test that leaves it changed makes a later test's
# kernels run on another device than its arrays, so test order would matter; the guards below fail it.
_DEVICE = str(wp.get_device())


def _left_changed() -> str | None:
    """The default device if it differs from the session's, which this puts back; else ``None``."""
    device = str(wp.get_device())
    if device == _DEVICE:
        return None
    wp.set_device(_DEVICE)
    return device


@pytest.fixture(autouse=True)
def _warp_device_guard():
    """Fail a test that leaves the Warp default device changed."""
    yield
    if left := _left_changed():
        pytest.fail(f"the test left the Warp default device on {left}, not {_DEVICE}: scope it with wp.ScopedDevice")


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    """Fail a test file whose import changes the Warp default device."""
    report = yield
    if isinstance(collector, pytest.Module):
        left = _left_changed()  # put the device back even when the import failed
        if left and report.passed:  # a failed import keeps its own traceback
            report.outcome = "failed"
            report.longrepr = (
                f"{collector.path.name} sets the Warp default device to {left} at import: scope it in its tests"
            )
    return report


@pytest.fixture
def warp_cpu():
    """Run the test with ``cpu`` as the Warp default device, and put the default back after it."""
    with wp.ScopedDevice("cpu"):
        yield


@pytest.fixture
def torchscript_policy(tmp_path):
    """Write a small TorchScript policy, ``obs[obs_dim] -> action[4]`` in [-1, 1], under ``tmp_path``.

    A stand-in for an exported ``policy.pt``: the same interface the goto_policy example deploys,
    with fixed random weights, so a test can fly it without training one.
    """
    torch = pytest.importorskip("torch")

    def _write(obs_dim: int = 16, name: str = "policy.pt") -> str:
        torch.manual_seed(0)
        net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64), torch.nn.Tanh(), torch.nn.Linear(64, 4), torch.nn.Tanh()
        )
        path = tmp_path / name
        torch.jit.script(net).save(str(path))
        return str(path)

    return _write


@pytest.fixture(scope="session")
def rrd_entities():
    """Read the entity paths a written ``.rrd`` carries, through the bundled command-line tool.

    rerun 0.34 dropped the local dataframe API, ``rerun.recording.load_recording``; reading an rrd
    now needs the catalog server + the optional datafusion dep. The version-matched command-line
    tool still prints per-chunk entity paths, so tests parse those.
    """

    def _entities(path: str) -> list[str]:
        out = subprocess.run(
            [sys.executable, "-m", "rerun", "rrd", "print", path], capture_output=True, text=True, check=True
        ).stdout
        return sorted(set(re.findall(r" - (/\S*) - data columns:", out)))

    return _entities


@pytest.fixture(scope="session")
def rrd_rows():
    """Read the rows a written ``.rrd`` holds, through Rerun's reader.

    One entry per chunk: its entity path, its static flag, and its component columns by name,
    each an Arrow array with one value per row.
    """

    def _rows(path: str) -> list[tuple[str, bool, dict]]:
        from rerun.experimental import RrdReader

        rows = []
        for chunk in RrdReader(path).stream():
            batch = chunk.to_record_batch()
            columns = {name: batch.column(name) for name in batch.schema.names if ":" in name}
            rows.append((chunk.entity_path, chunk.is_static, columns))
        return rows

    return _rows


class Layout:
    """The layout a recording stores: its views and containers, each by its display name."""

    def __init__(self, nodes: dict[str, dict]):
        self._nodes = nodes

    def _path(self, name: str) -> str:
        (path,) = [p for p, node in self._nodes.items() if node.get("display_name") == [name]]
        return path

    def view(self, name: str) -> dict:
        """The view named ``name``: its origin, its content rules, the entity its eye tracks and, by
        column kind, whether its log columns show.
        """
        path = self._path(name)
        columns = self._nodes.get(f"{path}/TextLogColumns", {}).get("text_log_columns", [])
        return {
            "origin": _rooted(self._nodes[path]["space_origin"][0]),
            "contents": self._nodes.get(f"{path}/ViewContents", {}).get("query", []),
            "eye": self._nodes.get(f"{path}/EyeControls3D", {}).get("tracking_entity", [None])[0],
            "columns": {column["kind"]: column["visible"] for column in columns},
        }

    def hidden(self, name: str) -> set[str]:
        """The paths the view named ``name`` leaves out by a rule of its own, each as the rule states it."""
        return {rule[1:].strip() for rule in self.view(name)["contents"] if rule.startswith("-")}

    def shows(self, name: str, entity: str) -> bool:
        """Whether the view named ``name`` takes in ``entity``, by Rerun's rule for a view's contents:
        the longest rule that matches the entity decides, and a rule ending in ``/**`` matches a subtree.
        """
        view = self.view(name)
        verdict, length = False, -1
        for rule in view["contents"]:
            path = rule.lstrip("+- ").replace("$origin", view["origin"].rstrip("/"))
            base = _rooted(path.removesuffix("/**"))
            subtree = path.endswith("/**") and (base == "/" or entity.startswith(f"{base}/"))
            if (entity == base or subtree) and len(base) > length:
                verdict, length = not rule.startswith("-"), len(base)
        return verdict

    def origins(self, name: str) -> list[str]:
        """The origin of every view at or under the tab named ``name``."""

        def walk(path: str) -> list[str]:
            node = self._nodes[path]
            if "space_origin" in node:
                return [_rooted(node["space_origin"][0])]
            return [origin for child in node.get("contents", []) for origin in walk(f"/{child}")]

        return walk(self._path(name))


def _rooted(path: str) -> str:
    """``path`` with one leading slash and no trailing one: the one form the tests compare."""
    return "/" + path.strip("/")


@pytest.fixture(scope="session")
def rrd_layout():
    """Read the layout a written ``.rrd`` stores, the last one its run sent, through Rerun's reader."""

    def _layout(path: str) -> Layout:
        import pyarrow as pa
        from rerun.experimental import RrdReader

        reader = RrdReader(path)
        sent = []  # one entry per stored layout: when the run sent it, and its nodes
        for store in reader.blueprints():
            nodes: dict[str, dict] = {}
            at = 0
            for chunk in reader.stream(store=store):
                batch = chunk.to_record_batch()
                if "log_time" in batch.schema.names:
                    at = max(at, *batch.column("log_time").cast(pa.int64()).to_pylist())
                node = nodes.setdefault(chunk.entity_path, {})
                for name in batch.schema.names:
                    if ":" in name:
                        node[name.split(":", 1)[1]] = batch.column(name).to_pylist()[-1]
            sent.append((at, nodes))
        return Layout(max(sent, key=lambda entry: entry[0])[1])

    return _layout


# The ticks the recorded flight steps: 0.4 s of sim time at 0.004 s a tick, which the Logger's 50 Hz
# log rate makes 20 logged ticks.
FLIGHT_TICKS = 100

# The sensors the recorded flight adds to the shipped camera vehicle, all on its base body: two
# cameras of one class, `A` and `B`, and a thermal camera, `Ir`. The run names each after its prim,
# in lower case.
FLIGHT_SENSORS = """\
over "astro_max"
{
    over "Geometry"
    {
        over "body_frd"
        {
            def Camera "A" (
                prepend apiSchemas = ["NexusCameraAPI"]
            )
            {
                int nexus:width = 64
                int nexus:height = 48
                float nexus:rate = 25
            }

            def Camera "B" (
                prepend apiSchemas = ["NexusCameraAPI"]
            )
            {
                int nexus:width = 64
                int nexus:height = 48
                float nexus:rate = 25
            }

            def Camera "Ir" (
                prepend apiSchemas = ["NexusThermalCameraAPI"]
            )
            {
                int nexus:width = 64
                int nexus:height = 48
                float nexus:rate = 25
            }
        }
    }
}
"""


class StandInReference:
    """A planned path through a mission's waypoints: what a tracking controller's planner returns."""

    duration = 1.0

    def __init__(self, waypoints):
        self._waypoints = [tuple(float(v) for v in w) for w in waypoints]
        self._start = (0.0, 0.0, 0.0)

    def set_start(self, position) -> None:
        self._start = tuple(float(v) for v in position)

    def reference_path(self) -> list[tuple[float, float, float]]:
        return [self._start, *self._waypoints]


class StandInController:
    """A controller named `standin` that commands nothing, takes a planned reference and logs one
    row of its own, `horizon`, each tick through the logger the loop hands it.
    """

    name = "standin"

    def __init__(self):
        self._logger = None
        self._controls = None

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def accept_setpoint(self, setpoint) -> None:
        pass

    def set_logger(self, logger) -> None:
        self._logger = logger

    def _act(self, tick) -> None:
        if self._controls is None:
            self._controls = wp.zeros((1, 16), dtype=wp.float32)  # every channel off: the vehicle rests
        tick.controls = self._controls
        if self._logger is not None:
            self._logger.log_strip("horizon", [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], color=(255, 140, 0))

    def stages(self):
        from nexus._src.core.interfaces import Stage

        return [Stage("act", "host", self._act)]


def fly_recorded(tmp, vehicle: str, *, layer: str | None = None, ticks: int = FLIGHT_TICKS, kit=None, debug=False):
    """Fly ``vehicle`` on the Warp CPU backend with recording on, and return the path of its ``.rrd``.

    The run flies a :class:`StandInController` under a guidance that plans a two-waypoint mission,
    with a ``Recorder`` attached and the Kit peer sent to its fake, or to ``kit``. ``layer`` is an
    override layer's text. The home folder is ``tmp`` for the run, so the recording lands there.
    """
    import nexus as na
    from nexus._src.build.assembly import build_orchestrator
    from nexus._src.build.launch import resolve_scenario
    from nexus._src.config import LaunchConfig
    from nexus._src.guidance import TrackingGuidance
    from nexus._src.peers.kit.fake import KitFake
    from nexus._src.rendering import rtx_renderer

    spec = {"vehicle": vehicle, "scene": "empty", "runtime": {"device": "cpu"}}
    if layer is not None:
        path = tmp / "layer.usda"
        path.write_text(f"#usda 1.0\n\n{layer}")
        spec["layer"] = str(path)
    with pytest.MonkeyPatch.context() as patch, wp.ScopedDevice("cpu"):
        patch.setenv("HOME", str(tmp))
        builder, _, cfg = resolve_scenario(LaunchConfig.from_dict(spec))
        controller = StandInController()
        renderer_factory = rtx_renderer(builder, cfg, peer=kit or KitFake)
        try:
            orch = build_orchestrator(
                "fixture",
                cfg,
                vehicle_builder=builder,
                controller=controller,
                rerun=True,
                viewer=False,
                debug=debug,
                renderer_factory=renderer_factory,
                max_steps=ticks,
            )
        except BaseException:
            if renderer_factory is not None:
                renderer_factory.close()  # the loop never took the Kit peer over, so it stops here
            raise
        guidance = TrackingGuidance(planner=StandInReference)
        with na.Sim.from_orchestrator(orch, guidance=guidance) as sim:
            sim.guidance.set_mission([(1.0, 0.0, 2.0), (2.0, 0.0, 2.0)])
            sim.run()
        return orch.logger.rrd_path


@pytest.fixture(scope="session")
def recorded_flight(tmp_path_factory) -> str:
    """The ``.rrd`` of one recorded flight of the shipped camera vehicle, `astro_max_fpv`, flown once
    for the session.

    The vehicle has mesh shapes, four rotor joints, an Inertial Measurement Unit (IMU) and the camera
    `FpvCam`, and the scene a ground plane. The flight adds ``FLIGHT_SENSORS`` and runs
    ``FLIGHT_TICKS`` ticks, as :func:`fly_recorded` flies it.
    """
    pytest.importorskip("rerun")
    pytest.importorskip("newton")
    return fly_recorded(tmp_path_factory.mktemp("flight"), "astro_max_fpv", layer=FLIGHT_SENSORS)
