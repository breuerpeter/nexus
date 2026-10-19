"""The public container peer: a project's own container peer subclasses it and starts, stops and reads alive
through it alone, and builds its image from a folder of its own through the public image builder.

The docker daemon is the system boundary: a stand-in daemon records each container the peer runs, each removal
and each image build, and gives each live container a console. The run tests build the fixture vehicle in
``tests/usd/sensor_vehicle.py`` on the Warp CPU backend through ``Sim``, with a stand-in controller that waits
where a row needs a run that waits on the peer. Skipped without newton or pxr.
"""

import time
from pathlib import Path
from typing import ClassVar

import pytest
from docker.errors import DockerException, ImageNotFound, NotFound

import nexus_uav._src.peers.containers as containers
from nexus_uav import ContainerPeer, RunError, Sim, ensure_image

pytest.importorskip("newton")
pytest.importorskip("pxr")

from tests.usd import sensor_vehicle as sv

pytestmark = pytest.mark.usefixtures("warp_cpu")

CONSOLE = b"radio up\n"


class _Container:
    def __init__(self, daemon, name, labels=None):
        self._daemon = daemon
        self.name = name
        self.labels = dict(labels or {})
        self.status = "running"

    def reload(self):
        pass

    def logs(self, stream=False, **kwargs):
        return iter([CONSOLE]) if stream else CONSOLE

    def remove(self, force=False):
        self._daemon.live.pop(self.name, None)
        self._daemon.removed.append(self.name)


class _Daemon:
    """Records every container run, every removal and every image build, and holds the live containers.

    No image exists until a build makes it. Each live container prints ``CONSOLE`` once.
    """

    def __init__(self):
        self.runs: list[dict] = []
        self.removed: list[str] = []
        self.builds: list[str] = []
        self.live: dict[str, _Container] = {}
        daemon = self

        class Images:
            def get(self, tag):
                if tag in daemon.builds:
                    return object()
                raise ImageNotFound(f"no image {tag}")

        class Api:
            def build(self, tag=None, **kwargs):
                daemon.builds.append(tag)
                yield {"stream": f"built {tag}\n"}

        class Containers:
            def run(self, image, **kwargs):
                daemon.runs.append({"image": image, **kwargs})
                container = _Container(daemon, kwargs.get("name"), kwargs.get("labels"))
                if container.name is not None:
                    daemon.live[container.name] = container
                return container

            def get(self, name):
                if name in daemon.live:
                    return daemon.live[name]
                raise NotFound(f"no container {name}")

        self.images = Images()
        self.api = Api()
        self.containers = Containers()

    def seed(self, name: str) -> _Container:
        """A live container called `name`, as a run killed without its teardown leaves one."""
        self.live[name] = _Container(self, name)
        return self.live[name]


@pytest.fixture
def daemon(monkeypatch, tmp_path):
    """The stand-in daemon, with the home folder in the test's folder."""
    d = _Daemon()
    monkeypatch.setattr(containers, "client", lambda: d)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return d


def _radio(image_dir: Path | None = None) -> type:
    """A project's container peer, `radio`, on the public container peer and nothing under `nexus_uav._src`.

    It runs the image `image_dir` builds, else a fixed one, with the command `radio --serve`. The class keeps
    each instance the build made in `made`. Each call makes a class of its own, for one run.
    """

    class Radio(ContainerPeer):
        name = "radio"
        made: ClassVar[list] = []

        def __init__(self, *, instance: int, log_dir, **run):
            super().__init__(instance=instance, log_dir=log_dir)
            type(self).made.append(self)

        def run_options(self) -> dict:
            image = "example.invalid/radio:1" if image_dir is None else ensure_image(image_dir, "radio")
            return {"image": image, "command": ["radio", "--serve"]}

    return Radio


class _Waiting(sv.Controller):
    """A stand-in controller that never attaches: the run waits on its peer, and a dead peer fails the wait."""

    attached = False


def _console(path: Path, timeout: float = 5.0) -> str:
    """The text at `path` once the log pump has written the console, else what the file holds at the timeout."""
    end = time.monotonic() + timeout
    while time.monotonic() < end and not (path.exists() and path.read_bytes() == CONSOLE):
        time.sleep(0.01)
    return path.read_text() if path.exists() else ""


def _run(tmp_path: Path, radio: type, controller: type = sv.Controller) -> Sim:
    """A run of the fixture vehicle that declares the stand-in peer on its scope `Companion`, mapped to `radio`."""
    registry = sv.components(StandInPeerAPI=radio, NexusPx4API_1=controller)
    vehicle = sv.vehicle(tmp_path, root=sv.companion())
    return Sim(vehicle, scene=sv.SCENE, device="cpu", registry=registry, run_dir=tmp_path / "run")


def test_a_projects_container_peer_subclasses_the_public_container_peer_and_starts_through_it_alone(daemon, tmp_path):
    """A project's container peer subclasses the public container peer and starts through it alone.

    Given a package whose peer class subclasses the public container peer, imports nothing under
    `nexus_uav._src`, states its image, command, name and label, and a `Registry` row for its schema, on the
    stand-in docker daemon, when a run that declares it builds, then the daemon ran the container under that
    name and label and the peer's console streams into `radio/console.log` in the run's folder.
    """
    radio = _radio()

    with _run(tmp_path, radio) as sim:
        sim.start(timeout=5.0)
        run = daemon.runs[0] if daemon.runs else {}
        console = _console(tmp_path / "run" / "radio" / "console.log")

    seen = (run.get("image"), run.get("command"), "radio" in str(run.get("name")), run.get("labels"), console)
    assert seen == ("example.invalid/radio:1", ["radio", "--serve"], True, {"nexus.peer": "radio"}, "radio up\n")


def test_a_projects_peer_stops_its_container_at_the_runs_end_and_a_second_stop_raises_nothing(daemon, tmp_path):
    """A project's peer stops its container at the run's end, and a second stop raises nothing.

    Given that peer on the stand-in daemon, when the run closes, then the daemon removed the container by
    name; when the peer stops again, nothing raises.
    """
    radio = _radio()

    with _run(tmp_path, radio) as sim:
        sim.start(timeout=5.0)
    removed_at_close = list(daemon.removed)
    again = [peer.stop() for peer in radio.made]

    names = [peer.container for peer in radio.made]
    assert (removed_at_close, again, daemon.removed) == (names, [None], names)


def test_a_projects_peer_reads_alive_from_the_daemon(daemon, tmp_path):
    """A project's peer reads alive from the daemon.

    Given that peer, when its container runs, then `alive` is true; when the daemon reports it exited,
    then `alive` is false and the run fails naming the peer.
    """
    radio = _radio()

    with pytest.raises(RunError) as failed, _run(tmp_path, radio, controller=_Waiting) as sim:
        peer = radio.made[0]
        while_running = peer.alive()
        for container in daemon.live.values():
            container.status = "exited"
        once_exited = peer.alive()
        sim.start(timeout=5.0)

    assert (while_running, once_exited, "radio exited" in str(failed.value)) == (True, False, True)


def test_a_leftover_container_of_the_peers_name_clears_before_its_new_one_starts(daemon, tmp_path):
    """A leftover container of the peer's name clears before its new one starts.

    Given the stand-in daemon seeded with a container under the project peer's name, when a run starts the
    peer, then the daemon removed the leftover and ran the new container.
    """
    peer = _radio()(instance=3, log_dir=tmp_path)
    daemon.seed(peer.container)

    peer.start()

    assert (daemon.removed, [run.get("name") for run in daemon.runs]) == ([peer.container], [peer.container])


def test_a_projects_image_builds_from_its_own_folder_once_per_machine_and_again_when_the_folder_changes(
    daemon, tmp_path
):
    """A project's image builds from its own folder once per machine and again when the folder changes.

    Given a project's peer that states a build folder, on the stand-in daemon lacking its image, when two runs
    start the peer, then the daemon built once; when a file in the folder changes and a third run starts, the
    daemon built again under a new tag.
    """
    image_dir = tmp_path / "image"
    image_dir.mkdir()
    (image_dir / "Dockerfile").write_text("FROM scratch\n")
    radio = _radio(image_dir)

    for instance in (1, 2):
        radio(instance=instance, log_dir=tmp_path).start()
    after_two = list(daemon.builds)
    (image_dir / "Dockerfile").write_text("FROM scratch\nLABEL changed=1\n")
    radio(instance=3, log_dir=tmp_path).start()

    tags = daemon.builds
    assert (len(after_two), len(tags), len(set(tags)), all(t.startswith("radio:") for t in tags)) == (1, 2, 2, True)


def test_a_projects_peer_start_with_no_reachable_daemon_fails_naming_the_daemons_address(monkeypatch, tmp_path):
    """A project's peer start with no reachable daemon fails naming the daemon's address.

    Given a daemon client that raises on ping, when the peer starts, then the run raises naming the daemon's
    address and no container ran.
    """
    import docker

    pinged: list[str] = []

    class Unreachable:
        def ping(self):
            pinged.append("ping")
            raise DockerException("no daemon")

    monkeypatch.setattr(docker, "from_env", lambda: Unreachable())
    monkeypatch.setattr(containers, "_client", None)  # no client this process reached earlier
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    radio = _radio()

    with pytest.raises(RunError) as failed, _run(tmp_path, radio) as sim:
        sim.start(timeout=5.0)

    message = str(failed.value)
    assert ("radio" in message, "unix:///var/run/docker.sock" in message, pinged) == (True, True, ["ping"])


def test_the_peer_label_holds_over_a_subclasss_own_labels(daemon, tmp_path):
    """The container carries `nexus.peer=<name>` whatever labels the subclass's run options add.

    Given a project's peer whose run options set `nexus.peer` to another value and a label of its own, when it
    starts, then the daemon ran the container with `nexus.peer=radio` and the peer's own label.
    """

    class Labelled(_radio()):
        def run_options(self) -> dict:
            return {**super().run_options(), "labels": {"nexus.peer": "other", "radio.band": "uhf"}}

    Labelled(instance=1, log_dir=tmp_path).start()

    assert daemon.runs[0].get("labels") == {"nexus.peer": "radio", "radio.band": "uhf"}
