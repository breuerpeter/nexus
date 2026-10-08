"""The PX4 peer as the run realizes it: managed, a container the build starts and the run's close
stops; external, an autopilot the run waits for on its own Hardware In The Loop (HIL) port; the
ports the run hands the peer; the peer's console log in the run's artifacts; and the px4-sitl image
built once from the Dockerfile the package ships.

The docker daemon and the autopilot on the far end of the HIL link are the system boundaries: a
stand-in daemon records what the run asks of it, and a stand-in autopilot dials the HIL port and
speaks MAVLink lockstep. The core assembly needs a full vehicle Universal Scene Description (USD)
file with motors and sensors, so the build's assembly step returns the loop over stand-in core
components, as ``tests/build/test_launch.py`` does.
"""

import hashlib
import json
import logging
import os
import socket
import threading
import time
from pathlib import Path

import pytest
import warp as wp
import yaml
from docker.errors import ImageNotFound, NotFound

# Pin the MAVLink dialect before importing mavutil, as the PX4 controller does: whichever import runs
# first sets it, and the stand-in autopilot must parse the controller's MAVLink 2 frames.
os.environ.setdefault("MAVLINK20", "1")
os.environ.setdefault("MAVLINK_DIALECT", "common")

from pymavlink import mavutil

import nexus_sim
import nexus_sim._src.build.launch as launch_mod
import nexus_sim._src.peers.containers as containers
from nexus_sim._src.api.sim import Sim
from nexus_sim._src.config import Catalog, LaunchConfig
from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.core.orchestrator import Orchestrator
from nexus_sim._src.core.schema import Controls, SimTime
from nexus_sim._src.core.signals import Signal
from nexus_sim._src.peers.px4_sitl.fake import Px4Fake
from nexus_sim._src.vehicle.controllers.px4 import controller as ctrl
from tests.usd import sensor_vehicle as sv

# --- the stand-in docker daemon -------------------------------------------------------------------


class _Container:
    def __init__(self, daemon, name, labels=None):
        self._daemon = daemon
        self.name = name
        self.labels = dict(labels or {})

    def logs(self, **kwargs):
        yield from ()

    def remove(self, force=False):
        self._daemon.live.pop(self.name, None)
        self._daemon.removed.append(self.name)


class _Daemon:
    """Records every container run, every removal and every image build.

    No image exists until the run builds it. A foreground run, ``detach=False``, returns an empty
    build log, as a completed container's output.
    """

    def __init__(self):
        self.runs: list[dict] = []
        self.removed: list[str] = []
        self.builds: list[dict] = []
        self.live: dict[str, _Container] = {}
        daemon = self

        class Images:
            def get(self, tag):
                if tag in {b["tag"] for b in daemon.builds}:
                    return object()
                raise ImageNotFound(f"no image {tag}")

        class Api:
            def build(self, **kwargs):
                daemon.builds.append(kwargs)
                yield {"stream": f"built {kwargs.get('tag')}\n"}

        class Containers:
            def run(self, image, **kwargs):
                daemon.runs.append({"image": image, **kwargs})
                if not kwargs.get("detach", True):
                    return b""
                container = _Container(daemon, kwargs.get("name"), kwargs.get("labels"))
                if container.name is not None:
                    daemon.live[container.name] = container
                return container

            def get(self, name):
                if name in daemon.live:
                    return daemon.live[name]
                raise NotFound(f"no container {name}")

            def list(self, **kwargs):
                wanted = dict(f.split("=", 1) for f in (kwargs.get("filters") or {}).get("label", []))
                return [c for c in daemon.live.values() if wanted.items() <= c.labels.items()]

        self.images = Images()
        self.api = Api()
        self.containers = Containers()


@pytest.fixture
def daemon(monkeypatch, tmp_path):
    """The stand-in daemon, a PX4 checkout stand-in at ``$PX4_DIR`` so nothing fetches, and a home
    folder in the test's folder.
    """
    d = _Daemon()
    monkeypatch.setattr(containers, "client", lambda: d)
    px4 = tmp_path / "px4"
    px4.mkdir()
    (px4 / "Makefile").write_text("px4_sitl:\n")  # what marks a folder as a PX4 tree
    monkeypatch.setenv("PX4_DIR", str(px4))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("NEXUS_ASSET_CACHE", str(tmp_path / "assets"))
    return d


# --- the stand-in core components, as tests/core builds the loop -----------------------------------


class _Clock:
    dt = 0.004

    def __init__(self):
        self._t = SimTime()

    def now(self):
        return self._t

    def advance(self):
        self._t = SimTime(self._t.sim_time + self.dt, self._t.step_index + 1)
        return self._t

    def throttle(self):
        pass


class _State:
    """One body at rest, level and nose north: the rows a stage that reads the vehicle's true state reads."""

    def __init__(self):
        self.body_q = wp.array(
            [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=wp.transform
        )  # forward-right-down body, up world
        self.body_qd = wp.zeros(1, dtype=wp.spatial_vector)


class _Physics:
    base_body = "body_frd"

    def reset(self):
        return _State()

    def clear_forces(self, state):
        pass

    def step(self, state, dt):
        return state

    def stages(self):
        return [
            Stage("clear", "device", lambda tick: self.clear_forces(tick.state)),
            Stage("step", "device", lambda tick: self.step(tick.state, tick.dt)),
        ]


class _Actuator:
    def forces(self, controls, state):
        pass

    def stages(self):
        return [Stage("forces", "device", lambda tick: self.forces(None, tick.state))]


def _loop(controller, **kw):
    return Orchestrator(
        clock=_Clock(),
        physics=_Physics(),
        actuator=_Actuator(),
        sensors=[],
        controller=controller,
        exchange_timeout=0.5,
        **kw,
    )


@pytest.fixture
def assembly(monkeypatch):
    """The build's assembly step returns the loop over the stand-in components, around the controller
    and the peers the build made.
    """

    def assemble(label, cfg, **kw):
        peers = {"peers": kw["peers"]} if "peers" in kw else {}
        return _loop(kw["controller"], preroll_timeout=kw.get("preroll_timeout", 2.0), **peers)

    monkeypatch.setattr(launch_mod, "build_orchestrator", assemble)


# --- a catalog with one vehicle, a local USD file --------------------------------------------------


def _vehicle(tmp_path) -> dict:
    blob = tmp_path / "vehicle.usda"
    blob.write_text(
        '#usda 1.0\n(\n    defaultPrim = "vehicle"\n)\n\n'
        'def Xform "vehicle" (\n    prepend apiSchemas = ["NexusPx4API", "NexusPx4SitlAPI"]\n)\n'
        '{\n    string nexus:airframe = "astro_max"\n}\n'
    )
    sha = hashlib.sha256(blob.read_bytes()).hexdigest()
    return {"astro": {"usd": {"url": blob.as_uri(), "sha256": sha, "filename": "vehicle.usda"}}}


@pytest.fixture
def catalog(tmp_path) -> Catalog:
    return Catalog.from_dict({"vehicles": _vehicle(tmp_path), "scenes": {"empty": {}}})


@pytest.fixture
def run(daemon, assembly, catalog, tmp_path):
    """Build a PX4 run of the catalog's vehicle, over the override layer given, if one is, and return its loop."""

    def _run(layer: str | None = None) -> Orchestrator:
        spec = {"vehicle": "astro", "scene": "empty"}
        if layer is not None:
            path = tmp_path / "override.usda"
            path.write_text(layer)
            spec["layer"] = str(path)
        launch = LaunchConfig.from_dict(spec)
        return launch_mod.build_from_launch(launch, catalog=catalog, cache_dir=tmp_path / "cache", preroll_timeout=1.0)

    return _run


# --- the network helpers ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _dial(port: int, within: float = 5.0) -> socket.socket | None:
    """A socket connected to the port, dialing again until something listens, or ``None``."""
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=1.0)
        except OSError:
            time.sleep(0.02)
    return None


class _Autopilot(threading.Thread):
    """An autopilot on the far end of the HIL link: it dials the port, answers each ``HIL_SENSOR``
    with ``HIL_ACTUATOR_CONTROLS`` for ``ticks`` ticks, then hangs up.
    """

    def __init__(self, port: int, ticks: int):
        super().__init__(name="px4-stand-in", daemon=True)
        self.port = port
        self.ticks = ticks
        self.answered = 0

    def run(self):
        sock = _dial(self.port, within=10.0)
        if sock is None:
            return
        sock.settimeout(5.0)
        mav = mavutil.mavlink.MAVLink(sock.makefile("wb", buffering=0), srcSystem=1, srcComponent=1)
        try:
            while self.answered < self.ticks:
                for msg in mav.parse_buffer(sock.recv(4096)) or []:
                    if msg.get_type() == "HIL_SENSOR" and self.answered < self.ticks:
                        mav.hil_actuator_controls_send(msg.time_usec, [0.0] * 16, 0, 0)
                        self.answered += 1
        except OSError:
            pass
        finally:
            sock.close()


# --- the rows ------------------------------------------------------------------------------------


def test_a_managed_peer_starts_when_the_run_starts_and_stops_when_the_run_closes(daemon, run):
    """A managed peer starts when the run starts and stops when the run closes.

    Given a PX4 run with a managed peer and a stand-in docker daemon, when the run enters and exits,
    then the daemon records one container start before the controller connects and one stop at close.
    """
    loop = run()
    started = [r["name"] for r in daemon.runs if r.get("detach", True)]  # the build's own foreground make aside

    loop.close()  # entered and left without a step: the controller never connected

    assert (len(started), daemon.removed) == (1, started), f"started {started}, removed {daemon.removed}"


def test_two_managed_runs_in_one_process_on_two_instances_keep_both_containers(daemon, run):
    """Two runs in one process take two instances: the second run's start removes nothing of the
    first, and each run's close removes only its own container.
    """
    first = run()
    second = run()
    started = [r["name"] for r in daemon.runs if r.get("detach", True)]

    first.close()
    removed_by_first = list(daemon.removed)
    second.close()

    assert (len(set(started)), removed_by_first, daemon.removed) == (2, [started[0]], started)


def test_a_peer_that_dies_ends_the_run_as_the_controllers_disconnect(daemon, caplog):
    """A peer that dies ends the run with an error that names the peer.

    Given a PX4 run whose peer is external, when the autopilot's link closes mid-run, then the run
    stops with ``ConnectionError`` naming PX4 and ``step()`` returns ``False`` after it.
    """
    caplog.set_level(logging.INFO, logger="nexus")
    port = _free_port()
    loop = _loop(ctrl.Px4MavlinkController(port=port), preroll_timeout=10.0)
    autopilot = _Autopilot(port, ticks=5)
    autopilot.start()

    ended = False
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if not loop.step():
            ended = True
            break
    autopilot.join(timeout=5.0)

    assert (ended, autopilot.answered, "Px4MavlinkController disconnected" in caplog.text) == (True, 5, True), (
        caplog.text
    )


def test_the_peers_ports_come_from_the_run(daemon, run):
    """The peer's ports come from the run, not from the peer.

    PX4 Software In The Loop (SITL) numbers every port from its instance, so the run hands the peer
    the instance it picked. Given a run and a stand-in docker daemon, when the peer starts, then the
    container's command carries an instance number and no port literal.
    """
    loop = run()

    launches = [r for r in daemon.runs if r.get("detach", True)]
    request = json.dumps([[r.get("command"), r.get("environment")] for r in launches], default=str)
    loop.close()

    commands = [r["command"] for r in launches if "-i" in r["command"]]
    numbered = [c[c.index("-i") + 1].isdigit() for c in commands]
    literals = [p for p in ("4560", "14540", "14550") if p in request]
    assert (len(launches), numbered, literals) == (1, [True], []), request


def test_the_peers_console_log_stays_in_the_runs_artifacts(daemon, assembly, catalog, tmp_path, monkeypatch):
    """The peer's console log stays in the run's artifacts.

    Given a PX4 run with a managed peer, when the run closes, then its artifacts list the peer's
    console log path.
    """
    project = tmp_path / "nexus.catalog.yaml"
    project.write_text(yaml.safe_dump({"vehicles": _vehicle(tmp_path)}))

    with Sim("astro", scene="empty", catalog=str(project), device="cpu", observe=False) as sim:
        pass

    log = Path(sim.artifacts()["px4_log"]).name
    assert log.startswith("px4-") and log.endswith(".log"), log


def test_the_px4_sitl_image_builds_once_from_the_packages_dockerfile(daemon, run):
    """The px4-sitl image builds locally from the Dockerfile the wheel ships, tagged by its hash, once.

    Given a stand-in docker daemon with no image under the tag, when the run enters, then the daemon
    receives one build of the package's Dockerfile at that tag; given the image exists, no build.
    """
    run().close()
    first = daemon.runs[0]["image"]
    run().close()

    built = [(b["tag"], Path(b["path"])) for b in daemon.builds]
    package = Path(nexus_sim.__file__).resolve().parent
    assert [
        (tag, (path / "Dockerfile").is_file() and path.resolve().is_relative_to(package)) for tag, path in built
    ] == [(first, True)], (built, first)


# --- the airframe the vehicle declares -------------------------------------------------------------


def _started_models(daemon) -> list[str]:
    """The `PX4_SIM_MODEL` of each PX4 container the run started, its foreground build aside."""
    return [r["environment"].get("PX4_SIM_MODEL") for r in daemon.runs if r.get("detach", True)]


def test_a_shipped_vehicle_flies_px4_sitl_on_its_own_airframe_with_no_control_flag(
    daemon, assembly, monkeypatch, tmp_path
):
    """A shipped vehicle flies PX4 SITL on its own airframe with no control flag.

    Given `astro_max_base` and a stand-in docker daemon, when the run builds, then PX4 SITL starts with
    `PX4_SIM_MODEL=none_astro_max`.
    """
    monkeypatch.delenv("NEXUS_ASSET_CACHE")  # the shipped vehicle comes from the checkout's own cache
    monkeypatch.chdir(tmp_path)  # no project catalog: only the bundled one

    with Sim("astro_max_base", scene="empty", device="cpu", observe=False):
        pass

    assert _started_models(daemon) == ["none_astro_max"]


def test_a_local_vehicle_usd_flies_the_airframe_its_px4_schema_declares(daemon, assembly, catalog, tmp_path):
    """A local vehicle USD flies the airframe its PX4 schema declares.

    Given a local vehicle USD whose PX4 schema declares airframe `foo` and a catalog whose default
    vehicle is another, when a managed run builds, then PX4 SITL starts with `PX4_SIM_MODEL=none_foo`.
    """
    usd = tmp_path / "local_vehicle.usda"
    usd.write_text(
        '#usda 1.0\n(\n    defaultPrim = "vehicle"\n)\n\n'
        'def Xform "vehicle" (\n    prepend apiSchemas = ["NexusPx4API", "NexusPx4SitlAPI"]\n)\n'
        '{\n    string nexus:airframe = "foo"\n}\n'
    )
    launch = LaunchConfig.from_dict({"vehicle": str(usd), "scene": "empty"})

    launch_mod.build_from_launch(launch, catalog=catalog, cache_dir=tmp_path / "cache", preroll_timeout=1.0).close()

    assert _started_models(daemon) == ["none_foo"]


# --- the fake PX4: the class a peer mapping sends the PX4 SITL peer to -----------------------------

_FAKE_PX4 = {"px4_sitl": Px4Fake}


class _Commands:
    """An actuator that keeps the controls it reads each tick, the commands the controller received."""

    def __init__(self):
        self.seen: list[tuple[float, ...]] = []
        self.controls = Signal("controls", Controls, shape=(1, 16))

    def forces(self, controls, state):
        pass

    def stages(self):
        def keep(tick):
            self.seen.append(tuple(self.controls.read()[0].tolist()))

        return [Stage("forces", "host", keep, reads=(self.controls,))]


def _step(loop, ticks: int) -> list[bool]:
    """Step ``ticks`` ticks, stopping at the first that fails: stepping an ended run starts a new one."""
    stepped = []
    while len(stepped) < ticks and (not stepped or stepped[-1]):
        stepped.append(loop.step())
    return stepped


def _faked(catalog, tmp_path) -> Orchestrator:
    """Build the catalog's vehicle with the PX4 SITL peer sent to its fake."""
    launch = LaunchConfig.from_dict({"vehicle": "astro", "scene": "empty"})
    return launch_mod.build_from_launch(
        launch, catalog=catalog, cache_dir=tmp_path / "cache", preroll_timeout=1.0, peers=_FAKE_PX4
    )


def test_a_run_whose_peer_mapping_sends_the_px4_sitl_peer_to_its_fake_starts_no_process_and_needs_no_px4_tree(
    daemon, assembly, catalog, monkeypatch, tmp_path
):
    """A run whose peer mapping sends the PX4 SITL peer to its fake starts no process and needs no PX4 tree.

    Given a stand-in docker daemon, no PX4 checkout and no `PX4_DIR`, when a run of the default
    vehicle, built with a peer mapping that sends the PX4 SITL peer to its fake, steps 500 ticks, then the
    daemon records no container, no fetch or build runs, and every tick completes.
    """
    monkeypatch.delenv("PX4_DIR")
    # The network is a boundary: a fetch this run must not make fails at once on an unreachable proxy.
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.setenv(var, "http://127.0.0.1:9")
    loop = _faked(catalog, tmp_path)

    stepped = _step(loop, 500)
    loop.close()

    fetched = (tmp_path / "home" / ".cache" / "nexus" / "px4").exists()
    assert (daemon.runs, daemon.builds, fetched, stepped.count(True)) == ([], [], False, 500)


def test_the_fake_px4_receives_each_tick_and_the_gps_at_its_sub_rate(daemon, warp_cpu, tmp_path):
    """The fake PX4 answers each `HIL_SENSOR` over the same lockstep.

    Given a run of the local fixture vehicle with an analytic sensor of each kind and the fake PX4, when
    it steps 500 ticks, then the fake receives a `HIL_SENSOR` each tick, and `HIL_GPS` and
    `HIL_STATE_QUATERNION` at the 10 Hz sub-rate of the Global Positioning System (GPS): 2 s of sim time
    at 0.004 s a tick, so 20 of each, or 19 where the float clock lands a GPS tick one late.
    """
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
    launch = LaunchConfig.from_dict({"vehicle": vehicle, "scene": sv.SCENE, "runtime": {"device": "cpu"}})
    loop = launch_mod.build_from_launch(launch, preroll_timeout=10.0, peers=_FAKE_PX4)

    _step(loop, 500)
    received = dict(loop.peers[0].received)
    loop.close()

    # The seed pass before the first tick exchanges once more, so 501 reach the fake.
    gps, state = received.get("HIL_GPS", 0), received.get("HIL_STATE_QUATERNION", 0)
    assert (received.get("HIL_SENSOR"), gps in (19, 20), state == gps) == (501, True, True), received


def test_the_controller_gets_the_fake_px4s_hover_command_every_tick(daemon, catalog, monkeypatch, tmp_path):
    """The fake PX4 answers each `HIL_SENSOR` with the fixed hover command.

    Given a run with the fake PX4, when it steps 500 ticks, then the controller hands the loop the same
    command with thrust on it every tick.
    """
    commands = _Commands()

    def assemble(label, cfg, **kw):
        return Orchestrator(
            clock=_Clock(),
            physics=_Physics(),
            actuator=commands,
            sensors=[],
            controller=kw["controller"],
            exchange_timeout=0.5,
            preroll_timeout=kw.get("preroll_timeout", 2.0),
            peers=kw["peers"],
        )

    monkeypatch.setattr(launch_mod, "build_orchestrator", assemble)
    loop = _faked(catalog, tmp_path)

    _step(loop, 500)
    loop.close()

    ticks = commands.seen[-500:]
    assert (len(ticks), len(set(ticks)), any(v > 0 for v in (ticks[0] if ticks else ()))) == (500, 1, True), set(ticks)


def test_the_px4_controller_is_built_the_same_way_whichever_process_answers(
    daemon, assembly, catalog, run, caplog, tmp_path
):
    """The build makes the PX4 controller the same way whichever process answers.

    Given the default vehicle, when a run builds it with the PX4 SITL peer sent to its fake and again
    with the peer external, then both loops hold the same stages in the same order, as each run's stage
    plan line says, and the fake answered the first tick.
    """
    caplog.set_level(logging.INFO, logger="nexus")

    def plan(loop) -> tuple[bool, list[str]]:
        flew = loop.step()  # the plan logs before the preroll: the external run then times out waiting for PX4
        loop.close()
        lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("stage plan:")]
        caplog.clear()
        return flew, lines

    flew, fake = plan(_faked(catalog, tmp_path))
    _, external = plan(run(_DROP_PX4_SITL))

    assert (flew, len(fake), fake) == (True, 1, external)


def test_a_fake_px4_whose_link_dies_ends_the_run_as_a_real_peers_death_does(
    daemon, assembly, catalog, caplog, tmp_path
):
    """A fake PX4 whose link dies ends the run as a real peer's death does.

    Given a run whose fake PX4 stops after 100 ticks, when the run steps on, then it stops with
    `ConnectionError` naming PX4 and `step()` returns `False` after it.
    """
    caplog.set_level(logging.INFO, logger="nexus")
    loop = _faked(catalog, tmp_path)
    answered = _step(loop, 100).count(True)

    loop.peers[0].stop()
    ended = False
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if not loop.step():
            ended = True
            break
    after = loop.step()

    assert (answered, ended, after, "Px4MavlinkController disconnected" in caplog.text) == (100, True, False, True), (
        caplog.text
    )


def test_a_peer_mapping_with_an_unknown_key_fails_the_build_before_any_peer_starts(daemon, assembly, catalog, tmp_path):
    """A peer mapping that names a peer the build doesn't know fails the build, before any peer starts.

    Given a stand-in docker daemon and a peer mapping keyed `px4-sitl`, a typo of `px4_sitl`, when the
    run builds, then it raises `ValueError` naming the unknown key and the known ones, and the daemon
    records no container.
    """
    launch = LaunchConfig.from_dict({"vehicle": "astro", "scene": "empty"})

    try:
        launch_mod.build_from_launch(
            launch, catalog=catalog, cache_dir=tmp_path / "cache", preroll_timeout=1.0, peers={"px4-sitl": Px4Fake}
        ).close()
        message = "no error"
    except ValueError as exc:
        message = str(exc)

    named = all(name in message for name in ("px4-sitl", "px4_sitl", "kit"))
    assert (named, daemon.runs) == (True, []), message


# --- the PX4 SITL peer the vehicle declares, and a layer that drops it -----------------------------

# A vehicle root that declares the PX4 controller and the PX4 SITL peer.
_DECLARED = (
    '#usda 1.0\n(\n    defaultPrim = "vehicle"\n)\n\n'
    'def Xform "vehicle" (\n    prepend apiSchemas = ["NexusPx4API", "NexusPx4SitlAPI"]\n)\n'
    '{\n    string nexus:airframe = "astro_max"\n}\n'
)

# A layer that drops the PX4 SITL peer's declaration, so the run attaches to an autopilot started elsewhere.
_DROP_PX4_SITL = '#usda 1.0\n\nover "vehicle" (\n    delete apiSchemas = ["NexusPx4SitlAPI"]\n)\n{\n}\n'


# A vehicle that declares the PX4 controller and the PX4 SITL peer on its controller's scope.
_SCOPED = (
    '#usda 1.0\n(\n    defaultPrim = "vehicle"\n)\n\n'
    'def Xform "vehicle"\n{\n'
    '    def Scope "Controller" (\n        prepend apiSchemas = ["NexusPx4API", "NexusPx4SitlAPI"]\n    )\n'
    '    {\n        string nexus:airframe = "astro_max"\n    }\n}\n'
)

# A layer that drops the PX4 SITL peer's declaration from the controller's scope.
_DROP_PX4_SITL_FROM_SCOPE = (
    '#usda 1.0\n\nover "vehicle"\n{\n'
    '    over "Controller" (\n        delete apiSchemas = ["NexusPx4SitlAPI"]\n    )\n    {\n    }\n}\n'
)


def _declared_run(tmp_path, layer: str | None = None, vehicle: str = _DECLARED) -> Orchestrator:
    """Build a run of a catalog vehicle that declares the PX4 SITL peer, `vehicle`, over `layer` when the
    caller passes one.
    """
    blob = tmp_path / "declared.usda"
    blob.write_text(vehicle)
    sha = hashlib.sha256(blob.read_bytes()).hexdigest()
    catalog = Catalog.from_dict(
        {
            "vehicles": {"astro": {"usd": {"url": blob.as_uri(), "sha256": sha, "filename": blob.name}}},
            "scenes": {"empty": {}},
        }
    )
    spec = {"vehicle": "astro", "scene": "empty"}
    if layer is not None:
        path = tmp_path / "override.usda"
        path.write_text(layer)
        spec["layer"] = str(path)
    launch = LaunchConfig.from_dict(spec)
    return launch_mod.build_from_launch(launch, catalog=catalog, cache_dir=tmp_path / "cache", preroll_timeout=5.0)


def test_a_vehicle_that_declares_the_px4_sitl_peer_starts_it(daemon, monkeypatch, tmp_path, warp_cpu):
    """A vehicle that declares the PX4 SITL peer starts it.

    Given each vehicle of the bundled catalog and no layer, when the run builds with the PX4 SITL peer
    sent to its fake, and the Kit peer to its own, then one fake PX4 starts and receives `HIL_SENSOR`
    over the HIL link as the run steps.
    """
    from nexus_sim._src.config.catalog import load_catalog
    from nexus_sim._src.peers.kit.fake import KitFake

    monkeypatch.delenv("NEXUS_ASSET_CACHE")  # the shipped vehicles come from the checkout's own cache
    monkeypatch.chdir(tmp_path)  # no project catalog: only the bundled one
    # The network is a boundary: a PX4 fetch this run must not make fails at once on an unreachable proxy.
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.setenv(var, "http://127.0.0.1:9")
    names = list(load_catalog().vehicles)

    flown = {}
    for name in names:
        launch = LaunchConfig.from_dict({"vehicle": name, "scene": "empty", "runtime": {"device": "cpu"}})
        loop = launch_mod.build_from_launch(launch, preroll_timeout=10.0, peers={"px4_sitl": Px4Fake, "kit": KitFake})
        _step(loop, 10)
        fakes = [p for p in loop.peers if isinstance(p, Px4Fake)]
        flown[name] = (len(fakes), bool(fakes) and fakes[0].received["HIL_SENSOR"] > 0)
        loop.close()

    assert flown == dict.fromkeys(names, (1, True))


def test_a_layer_that_drops_the_px4_sitl_peer_starts_no_px4_and_waits_on_instance_0s_hil_port(
    daemon, assembly, tmp_path
):
    """A layer that drops the PX4 SITL peer starts no PX4 and waits on instance 0's HIL port.

    Given a stand-in docker daemon and a layer that drops the PX4 SITL peer, when the run enters and a
    fake PX4 the test starts dials instance 0's HIL port, 4560, then the daemon records no container
    and the run steps against the fake.
    """
    loop = _declared_run(tmp_path, _DROP_PX4_SITL)
    autopilot = Px4Fake(instance=0)
    autopilot.start()

    stepped = loop.step()
    loop.close()
    autopilot.stop()

    assert (daemon.runs, stepped) == ([], True)


def test_a_layer_that_drops_the_px4_sitl_peer_from_the_controllers_scope_flies_an_autopilot_started_elsewhere(
    daemon, assembly, tmp_path
):
    """A layer that drops the PX4 SITL peer from the controller's scope flies an autopilot started elsewhere.

    Given a vehicle whose `Scope` `/vehicle/Controller` declares PX4 and the PX4 SITL peer, and a layer
    whose `over "Controller"` under `/vehicle` deletes `NexusPx4SitlAPI`, when the run builds, then no
    peer starts and PX4 dials instance 0's HIL port, 4560: the daemon records no container, and the run
    steps against a fake PX4 the test starts there.
    """
    loop = _declared_run(tmp_path, _DROP_PX4_SITL_FROM_SCOPE, vehicle=_SCOPED)
    autopilot = Px4Fake(instance=0)
    autopilot.start()

    stepped = loop.step()
    loop.close()
    autopilot.stop()

    assert (daemon.runs, stepped) == ([], True)


def test_a_second_external_run_on_one_machine_fails_and_names_the_holder(daemon, assembly, tmp_path, caplog):
    """A second external run on one machine fails and names the holder.

    Given a process that holds instance 0's HIL port, 4560, as a first run does, when a run whose
    layer drops the PX4 SITL peer enters, then it fails and names the process that holds the port.
    """
    caplog.set_level(logging.INFO, logger="nexus")
    holder = socket.socket()
    holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    holder.bind(("127.0.0.1", 4560))
    holder.listen()

    failed, said = False, ""
    try:
        loop = _declared_run(tmp_path, _DROP_PX4_SITL)
        failed = not loop.step()
        loop.close()
    except Exception as exc:
        failed, said = True, str(exc)
    finally:
        holder.close()

    assert (failed, str(os.getpid()) in said + caplog.text) == (True, True), said + caplog.text


def test_a_managed_run_takes_a_free_px4_instance_itself(daemon, assembly, tmp_path):
    """A managed run takes a free PX4 instance itself.

    Given a stand-in docker daemon where a live PX4, owned by this process, holds instance 0, when a
    run of a vehicle that declares the PX4 SITL peer enters, then its PX4 container runs instance 1,
    `-i 1`, and the run listens on instance 1's HIL port, 4561.
    """
    from nexus_sim._src.peers import LABEL

    held = _Container(
        daemon, "nexus-px4-held-0", {LABEL: "px4", "nexus.px4.instance": "0", "nexus.owner": str(os.getpid())}
    )
    daemon.live[held.name] = held

    loop = _declared_run(tmp_path)
    commands = [r.get("command") for r in daemon.runs if r.get("detach", True)]
    stepping = threading.Thread(target=loop.step, daemon=True)  # the preroll waits for PX4 to dial in
    stepping.start()
    dialed = _dial(4561)
    stepping.join(timeout=15.0)
    loop.close()
    if dialed is not None:
        dialed.close()

    instances = [c[c.index("-i") + 1] for c in commands if c and "-i" in c]
    assert (instances, dialed is not None) == (["1"], True)


def test_a_run_skips_an_instance_another_runs_claim_holds(daemon, assembly, tmp_path):
    """A managed run takes a free PX4 instance itself, and two runs that start at once take two.

    Given another run's claim on instance 0, a lock held on its file under the home folder's
    `.cache/nexus/px4-instances/` and no container yet, when a run of a vehicle that declares the PX4
    SITL peer enters, then its PX4 container runs instance 1, `-i 1`.
    """
    import fcntl

    locks = tmp_path / "home" / ".cache" / "nexus" / "px4-instances"
    locks.mkdir(parents=True)
    with open(locks / "0.lock", "w") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        loop = _declared_run(tmp_path)
        commands = [r.get("command") for r in daemon.runs if r.get("detach", True)]
        loop.close()

    instances = [c[c.index("-i") + 1] for c in commands if c and "-i" in c]
    assert instances == ["1"]
