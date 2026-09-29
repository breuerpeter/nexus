"""The PX4 peer as the run realizes it: managed, a container the build starts and the run's close
stops; external, an autopilot the run waits for on its own Hardware In The Loop (HIL) port; the
ports the run hands the peer; the peer's console log in the run's artifacts; and the px4-sitl image
built once from the Dockerfile the package ships.

The docker daemon and the autopilot on the far end of the HIL link are the system boundaries: a
stand-in daemon records what the run asks of it, and a stand-in autopilot dials the HIL port and
speaks MAVLink lockstep. The core assembly needs a full vehicle Universal Scene Description (USD)
file with motors and sensors, so the build's assembly step returns the loop over stand-in core
components, as ``tests/runtimes/test_launch.py`` does.
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
import yaml
from docker.errors import ImageNotFound, NotFound

# Pin the MAVLink dialect before importing mavutil, as the PX4 controller does: whichever import runs
# first sets it, and the stand-in autopilot must parse the controller's MAVLink 2 frames.
os.environ.setdefault("MAVLINK20", "1")
os.environ.setdefault("MAVLINK_DIALECT", "common")

from pymavlink import mavutil

import nexus
import nexus._src.build.launch as launch_mod
import nexus._src.peers.containers as containers
from nexus._src.api.sim import Sim
from nexus._src.config import LaunchConfig, Registry
from nexus._src.core.interfaces import Stage
from nexus._src.core.orchestrator import Orchestrator
from nexus._src.core.schema import SimTime
from nexus._src.vehicle.controllers.px4 import controller as ctrl

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


class _Physics:
    base_body = "body_frd"

    def reset(self):
        return {"q": 0}

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
        return [Stage("forces", "device", lambda tick: self.forces(tick.controls, tick.state))]


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
        'def Xform "vehicle" (\n    prepend apiSchemas = ["NexusPx4API"]\n)\n{\n    string nexus:airframe = "astro_max"\n}\n'
    )
    sha = hashlib.sha256(blob.read_bytes()).hexdigest()
    return {"name": "astro", "usd": {"url": blob.as_uri(), "sha256": sha, "filename": "vehicle.usda"}}


@pytest.fixture
def catalog(tmp_path) -> Registry:
    return Registry.from_dict(
        {"vehicles": [_vehicle(tmp_path)], "scenes": {"empty": {}}, "defaults": {"vehicle": "astro", "scene": "empty"}}
    )


@pytest.fixture
def run(daemon, assembly, catalog, tmp_path):
    """Build a PX4 run whose ``peers.px4`` entry is the one given, and return its loop."""

    def _run(px4: dict) -> Orchestrator:
        launch = LaunchConfig.from_dict({"vehicle": "astro", "peers": {"px4": px4}})
        return launch_mod.build_from_launch(launch, registry=catalog, cache_dir=tmp_path / "cache", preroll_timeout=1.0)

    return _run


# --- the network helpers ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _free_instance() -> int:
    """A PX4 instance whose HIL port, 4560 + N, nothing on this machine holds."""
    for instance in range(20, 200):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", 4560 + instance))
            except OSError:
                continue
            return instance
    raise RuntimeError("no free PX4 instance")


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
    loop = run({"realization": "managed"})
    started = [r["name"] for r in daemon.runs if r.get("detach", True)]  # the build's own foreground make aside

    loop.close()  # entered and left without a step: the controller never connected

    assert (len(started), daemon.removed) == (1, started), f"started {started}, removed {daemon.removed}"


def test_two_managed_runs_in_one_process_on_two_instances_keep_both_containers(daemon, run):
    """Two runs in one process, instances 0 and 1: the second run's start removes nothing of the
    first, and each run's close removes only its own container.
    """
    first = run({"realization": "managed", "instance": 0})
    second = run({"realization": "managed", "instance": 1})
    started = [r["name"] for r in daemon.runs if r.get("detach", True)]

    first.close()
    removed_by_first = list(daemon.removed)
    second.close()

    assert (len(set(started)), removed_by_first, daemon.removed) == (2, [started[0]], started)


def test_an_external_peer_starts_no_process_and_the_run_waits_on_its_hil_port(daemon, run):
    """An external peer starts no process: the run listens on its HIL address and waits for the
    autopilot to dial in.

    Given a PX4 run with an external peer and a stand-in docker daemon, when the run enters, then the
    daemon records no container and the controller waits on the run's HIL port, 4560 + instance.
    """
    instance = _free_instance()
    port = 4560 + instance
    loop = run({"realization": "external", "instance": instance})

    stepping = threading.Thread(target=loop.step, daemon=True)  # the preroll waits for the autopilot
    stepping.start()
    dialed = _dial(port)
    stepping.join(timeout=15.0)
    if dialed is not None:
        dialed.close()

    assert (daemon.runs, dialed is not None) == ([], True)


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

    PX4 Software In The Loop (SITL) numbers every port from its instance, so the run names the instance. Given a run whose
    launch names its PX4 instance and a stand-in docker daemon, when the peer starts, then the
    container's command carries that instance and no port literal.
    """
    loop = run({"realization": "managed", "instance": 3})

    launches = [r for r in daemon.runs if r.get("detach", True)]
    request = json.dumps([[r.get("command"), r.get("environment")] for r in launches], default=str)
    loop.close()

    instance = ["-i", "3"] if any("-i" in r["command"] and "3" in r["command"] for r in launches) else []
    literals = [p for p in ("4560", "14540", "14550") if p in request]
    assert (len(launches), instance, literals) == (1, ["-i", "3"], []), request


def test_the_peers_console_log_stays_in_the_runs_artifacts(daemon, assembly, catalog, tmp_path, monkeypatch):
    """The peer's console log stays in the run's artifacts.

    Given a PX4 run with a managed peer, when the run closes, then its artifacts list the peer's
    console log path.
    """
    project = tmp_path / "nexus.registry.yaml"
    project.write_text(yaml.safe_dump({"vehicles": [_vehicle(tmp_path)]}))

    with Sim("astro", registry=str(project), device="cpu", observe=False) as sim:
        pass

    log = Path(sim.artifacts()["px4_log"]).name
    assert log.startswith("px4-") and log.endswith(".log"), log


def test_the_px4_sitl_image_builds_once_from_the_packages_dockerfile(daemon, run):
    """The px4-sitl image builds locally from the Dockerfile the wheel ships, tagged by its hash, once.

    Given a stand-in docker daemon with no image under the tag, when the run enters, then the daemon
    receives one build of the package's Dockerfile at that tag; given the image exists, no build.
    """
    run({"realization": "managed"}).close()
    first = daemon.runs[0]["image"]
    run({"realization": "managed"}).close()

    built = [(b["tag"], Path(b["path"])) for b in daemon.builds]
    package = Path(nexus.__file__).resolve().parent
    assert [
        (tag, (path / "Dockerfile").is_file() and path.resolve().is_relative_to(package)) for tag, path in built
    ] == [(first, True)], (built, first)


# --- the airframe the vehicle declares -------------------------------------------------------------


def _started_models(daemon) -> list[str]:
    """The `PX4_SIM_MODEL` of each PX4 container the run started, its foreground build aside."""
    return [r["environment"].get("PX4_SIM_MODEL") for r in daemon.runs if r.get("detach", True)]


class _Offboard:
    """The operator link's far end stands in for PX4: it answers at once, so the run needs no MAVLink peer."""

    connected = True

    def __init__(self, *args, **kwargs):
        pass

    def open(self):
        return self

    def close(self):
        pass


def test_a_shipped_vehicle_flies_px4_sitl_on_its_own_airframe_with_no_control_flag(
    daemon, assembly, monkeypatch, tmp_path
):
    """A shipped vehicle flies PX4 SITL on its own airframe with no control flag.

    Given `astro_max_base` and a stand-in docker daemon, when the run builds, then PX4 SITL starts with
    `PX4_SIM_MODEL=none_astro_max` and the operator is `Px4Offboard`, here its stand-in.
    """
    import nexus._src.operator as op_mod

    monkeypatch.delenv("NEXUS_ASSET_CACHE")  # the shipped vehicle comes from the checkout's own cache
    monkeypatch.chdir(tmp_path)  # no project catalog: only the bundled one
    monkeypatch.setattr(op_mod, "Px4Offboard", _Offboard)

    with Sim("astro_max_base", device="cpu", observe=False) as sim:
        operator = sim.operator

    assert (_started_models(daemon), type(operator)) == (["none_astro_max"], _Offboard)


def test_a_local_vehicle_usd_flies_the_airframe_its_px4_schema_declares(daemon, assembly, catalog, tmp_path):
    """A local vehicle USD flies the airframe its PX4 schema declares.

    Given a local vehicle USD whose PX4 schema declares airframe `foo` and a catalog whose default
    vehicle is another, when a managed run builds, then PX4 SITL starts with `PX4_SIM_MODEL=none_foo`.
    """
    usd = tmp_path / "local_vehicle.usda"
    usd.write_text(
        '#usda 1.0\n(\n    defaultPrim = "vehicle"\n)\n\n'
        'def Xform "vehicle" (\n    prepend apiSchemas = ["NexusPx4API"]\n)\n'
        '{\n    string nexus:airframe = "foo"\n}\n'
    )
    launch = LaunchConfig.from_dict({"vehicle": str(usd), "peers": {"px4": {"realization": "managed"}}})

    launch_mod.build_from_launch(launch, registry=catalog, cache_dir=tmp_path / "cache", preroll_timeout=1.0).close()

    assert _started_models(daemon) == ["none_foo"]


# --- the fake PX4, the realization that stands in for the process ---------------------------------


class _Commands:
    """An actuator that keeps the command it's handed each tick, the controls the controller received."""

    def __init__(self):
        self.seen: list[tuple[float, ...]] = []

    def forces(self, controls, state):
        pass

    def stages(self):
        return [Stage("forces", "host", lambda tick: self.seen.append(tuple(tick.controls.numpy()[0].tolist())))]


def test_a_run_that_picks_the_fake_px4_starts_no_process_and_needs_no_px4_tree(daemon, run, monkeypatch, tmp_path):
    """A run that picks the fake PX4 starts no process and needs no PX4 tree.

    Given a stand-in docker daemon, no PX4 checkout and no `PX4_DIR`, when a run of the default
    vehicle with its PX4 peer fake steps 500 ticks, then the daemon records no container, no fetch or
    build runs, and every tick completes.
    """
    monkeypatch.delenv("PX4_DIR")
    loop = run({"realization": "fake"})

    stepped = [loop.step() for _ in range(500)]
    loop.close()

    fetched = (tmp_path / "home" / ".cache" / "nexus" / "px4").exists()
    assert (daemon.runs, daemon.builds, fetched, stepped.count(True)) == ([], [], False, 500)


def test_the_fake_px4_receives_each_tick_and_the_gps_at_its_sub_rate(daemon, monkeypatch, warp_cpu):
    """The fake PX4 answers each `HIL_SENSOR` over the same lockstep.

    Given a run of `astro_max_base` with the fake PX4, when it steps 500 ticks, then the fake receives
    a `HIL_SENSOR` each tick, and `HIL_GPS` and `HIL_STATE_QUATERNION` at the 10 Hz sub-rate of the Global
    Positioning System (GPS): 2 s of sim time at 0.004 s a tick, so 20 of each, or 19 where the float
    clock lands a GPS tick one late.
    """
    monkeypatch.delenv("NEXUS_ASSET_CACHE")  # the shipped vehicle comes from the checkout's own cache
    launch = LaunchConfig.from_dict(
        {"vehicle": "astro_max_base", "runtime": {"device": "cpu"}, "peers": {"px4": {"realization": "fake"}}}
    )
    loop = launch_mod.build_from_launch(launch, preroll_timeout=10.0)

    for _ in range(500):
        loop.step()
    received = dict(loop.peers[0].received)
    loop.close()

    # The seed pass before the first tick exchanges once more, so 501 reach the fake.
    gps, state = received.get("HIL_GPS", 0), received.get("HIL_STATE_QUATERNION", 0)
    assert (received.get("HIL_SENSOR"), gps in (19, 20), state == gps) == (501, True, True), received


def test_the_controller_gets_the_fake_px4s_hover_command_every_tick(daemon, catalog, monkeypatch, tmp_path):
    """The fake PX4 answers each `HIL_SENSOR` with the fixed hover command.

    Given a run with the fake PX4, when it steps 500 ticks, then the controller hands the loop a
    command each tick, and every one is the same command, with thrust on it.
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
    launch = LaunchConfig.from_dict({"vehicle": "astro", "peers": {"px4": {"realization": "fake"}}})
    loop = launch_mod.build_from_launch(launch, registry=catalog, cache_dir=tmp_path / "cache", preroll_timeout=1.0)

    for _ in range(500):
        loop.step()
    loop.close()

    ticks = commands.seen[-500:]
    assert (len(ticks), len(set(ticks)), any(v > 0 for v in ticks[0])) == (500, 1, True), set(ticks)


def test_the_px4_controller_is_built_the_same_way_whichever_realization_answers(daemon, run, caplog):
    """The build makes the PX4 controller the same way whichever realization answers.

    Given the default vehicle, when a run builds it with the PX4 peer fake and again with it external,
    then both loops hold the same stages in the same order, as each run's stage plan line says.
    """
    caplog.set_level(logging.INFO, logger="nexus")

    def plan(px4: dict) -> list[str]:
        caplog.clear()
        loop = run(px4)
        loop.step()  # the plan logs before the preroll: the external run then times out waiting for PX4
        loop.close()
        return [r.getMessage() for r in caplog.records if r.getMessage().startswith("stage plan:")]

    fake = plan({"realization": "fake"})
    external = plan({"realization": "external", "instance": _free_instance()})

    assert (len(fake), fake) == (1, external)


def test_a_fake_px4_whose_link_dies_ends_the_run_as_a_real_peers_death_does(daemon, run, caplog):
    """A fake PX4 whose link dies ends the run as a real peer's death does.

    Given a run whose fake PX4 stops after 100 ticks, when the run steps on, then it stops with
    `ConnectionError` naming PX4 and `step()` returns `False` after it.
    """
    caplog.set_level(logging.INFO, logger="nexus")
    loop = run({"realization": "fake"})
    for _ in range(100):
        loop.step()

    loop.peers[0].stop()
    ended = False
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if not loop.step():
            ended = True
            break
    after = loop.step()

    assert (ended, after, "Px4MavlinkController disconnected" in caplog.text) == (True, False, True), caplog.text


def test_the_operator_on_a_fake_px4_run_fails_at_once_with_a_clear_error(daemon, assembly, catalog, tmp_path):
    """The operator on a fake PX4 run fails at once with a clear error.

    Given a run with the fake PX4, when a caller takes `sim.operator`, then it raises an error that names the
    fake PX4, and nothing waits on the offboard port.
    """
    project = tmp_path / "nexus.registry.yaml"
    project.write_text(yaml.safe_dump({"vehicles": [_vehicle(tmp_path)]}))

    with Sim("astro", registry=str(project), device="cpu", observe=False, px4="fake") as sim:
        sim.start(timeout=5.0)
        t0 = time.monotonic()
        try:
            message = f"no error: {sim.operator!r}"
        except RuntimeError as exc:
            message = str(exc)
        waited = time.monotonic() - t0

    assert ("fake" in message, waited < 5.0) == (True, True), (message, waited)
