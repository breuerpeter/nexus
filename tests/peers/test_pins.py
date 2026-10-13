"""Every peer resolves its pin the same way: the file the peer ships in its folder, a project's
``nexus.<peer>.ref`` beside its catalog over it, and the peer's variable, ``NEXUS_PX4_SITL_REF`` or
``NEXUS_KIT_REF``, over both; and
the run's receipt records each peer's pin and where it came from.

The docker daemon and the network are the system boundaries. A stand-in daemon records what a run asks
of it, a git repository in the test's folder stands in for a PX4-Autopilot fork on GitHub, and every
other fetch fails at once on a cut network.
"""

import importlib.resources
import subprocess
from pathlib import Path

import pytest
from docker.errors import ImageNotFound, NotFound

import nexus_uav._src.build.launch as L
import nexus_uav._src.peers.containers as containers
import nexus_uav._src.peers.kit.runner as kit_runner
from nexus_uav._src.config import LaunchConfig
from nexus_uav._src.peers.px4_sitl.fake import Px4Fake
from tests.usd import sensor_vehicle as sv

# PX4's ports are the machine's, so the tests that bind or dial them run on one worker, one at a time.
pytestmark = [pytest.mark.xdist_group("px4_ports"), pytest.mark.usefixtures("warp_cpu")]

# A camera on the base body, which the Kit peer renders, small so a frame would cost little.
CAMERA = sv.prim(
    "Cam", "NexusCameraAPI", "int nexus:width = 8\nint nexus:height = 6\nfloat nexus:rate = 25", kind="Camera"
)
FORK = "https://github.com/example/px4.git"  # the PX4 fork a pin names, which the stand-in repository answers for


class Daemon:
    """A stand-in docker daemon: every image exists unless ``held`` names the ones that do, a pull records
    its image in ``pulls`` and holds it, and a started container records its spec in ``runs``.
    """

    def __init__(self):
        self.runs: list[dict] = []
        self.pulls: list[str] = []
        self.held: set[str] | None = None
        daemon = self

        class Container:
            status = "running"

            def __init__(self, name):
                self.name = name

            def reload(self):
                pass

            def remove(self, force=False):
                self.status = "removed"

        class Images:
            def get(self, tag):
                if daemon.held is not None and tag not in daemon.held:
                    raise ImageNotFound(f"no image {tag}")
                return object()

        class Api:
            def pull(self, repository, tag=None, stream=False, **kwargs):
                ref = f"{repository}:{tag}" if tag else repository
                daemon.pulls.append(ref)
                if daemon.held is not None:
                    daemon.held.add(ref)
                return iter([{"status": f"Downloaded newer image for {ref}"}]) if stream else ""

        class Containers:
            def run(self, image, **kwargs):
                daemon.runs.append({"image": image, **kwargs})
                return b"" if not kwargs.get("detach", True) else Container(kwargs.get("name"))

            def get(self, name):
                raise NotFound(f"no container {name}")

            def list(self, **kwargs):
                return []

        self.images = Images()
        self.api = Api()
        self.containers = Containers()

    def started(self) -> list[dict]:
        """The containers a run started and left running: a peer's, not a build's."""
        return [run for run in self.runs if run.get("detach", True)]


@pytest.fixture
def daemon(monkeypatch, tmp_path, cut_network):
    """The stand-in daemon, a home folder in the test's folder for the peers' caches, no pin variable or
    ``PX4_DIR`` from the machine, and the network cut.
    """
    d = Daemon()
    monkeypatch.setattr(containers, "client", lambda: d)
    monkeypatch.setattr(kit_runner, "client", lambda: d)  # the Kit runner holds the name it imported
    monkeypatch.setattr(containers, "_pump_logs", lambda container, log_path: None)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for var in ("PX4_DIR", "NEXUS_PX4_SITL_REF", "NEXUS_KIT_REF"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # no project catalog unless a test writes one
    cut_network()
    return d


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _checkout(folder: Path) -> str:
    """A stand-in PX4 tree with one commit at ``folder``. Returns that commit's sha."""
    folder.mkdir(parents=True)
    (folder / "Makefile").write_text("px4_sitl:\n\ttrue\n")
    _git(folder, "init", "-q")
    _git(folder, "add", "-A")
    _git(folder, "commit", "-q", "-m", "a PX4 tree")
    _git(folder, "config", "uploadpack.allowAnySHA1InWant", "true")  # a fetch by sha, as from GitHub
    return _git(folder, "rev-parse", "HEAD")


def _fork(tmp_path, monkeypatch) -> str:
    """A stand-in for the fork ``example/px4`` on GitHub: git fetches from a repository in the test's
    folder in its place. Returns the sha of its one commit.
    """
    repo = tmp_path / "fork"
    sha = _checkout(repo)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", f"url.{repo}.insteadOf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", FORK)
    return sha


def _shipped(peer: str) -> Path:
    """The pin file the peer ships in its own folder."""
    return Path(str(importlib.resources.files(f"nexus_uav._src.peers.{peer}").joinpath("pin.ref")))


def _project(tmp_path, monkeypatch, **pins: str) -> Path:
    """A project folder with a catalog and the pin files ``pins`` names, filename to content, as the
    working folder. Returns the folder.
    """
    project = tmp_path / "project"
    project.mkdir()
    (project / "nexus.catalog.yaml").write_text("vehicles: {}\nscenes: {}\n")
    for name, text in pins.items():
        (project / name).write_text(text)
    monkeypatch.chdir(project)
    return project


def _run(tmp_path, body: str = "", *, px4: bool = True, peers: dict | None = None) -> dict:
    """Build the fixture vehicle, with ``body`` under its base body and with the PX4 Software In The Loop (SITL) peer declared
    when ``px4``, in the fixture scene. Returns what the run left: its receipt as JSON, or the error the
    build raised, which ends a run as well.
    """
    launch = LaunchConfig.from_dict(
        {"vehicle": sv.vehicle(tmp_path, body, px4=px4), "scene": sv.SCENE, "runtime": {"device": "cpu"}}
    )
    try:
        loop = L.build_from_launch(launch, preroll_timeout=10.0, peers=peers)
    except Exception as exc:
        return {"receipt": {}, "error": exc}
    loop.close()
    return {"receipt": loop.settings, "error": None}


def _pin(outcome: dict, peer: str) -> dict | None:
    return outcome["receipt"].get("peers", {}).get(peer)


def _px4_tree(daemon: Daemon) -> str | None:
    """The PX4 tree the run's PX4 container flies: the folder it works in."""
    started = daemon.started()
    return started[0]["working_dir"] if started else None


def test_a_px4_run_with_no_project_pin_flies_the_shipped_pin_and_records_it(daemon, tmp_path):
    """A PX4 run with no project pin file and no variable flies the PX4 peer's shipped pin, and the
    receipt records that pin and the shipped file.

    Given the stand-in daemon and the shipped pin's tree already on this machine, when a PX4 run builds
    on the bundled catalog, then PX4 flies that tree, and the receipt's PX4 pin is the shipped file's
    `repo@sha` with that file as its source.
    """
    shipped = _shipped("px4_sitl")
    ref = shipped.read_text().splitlines()[0].strip() if shipped.is_file() else "PX4/PX4-Autopilot@" + "0" * 40
    tree = tmp_path / "home" / ".cache" / "nexus" / "px4" / ref.split("@", 1)[1]
    tree.mkdir(parents=True)
    (tree / "Makefile").write_text("px4_sitl:\n\ttrue\n")

    outcome = _run(tmp_path)

    assert (_px4_tree(daemon), _pin(outcome, "px4_sitl")) == (
        str(tree),
        {"ref": ref, "source": str(shipped), "commit": None},
    ), outcome["error"]


def test_a_projects_px4_pin_file_overrides_the_shipped_pin_and_the_receipt_names_it(daemon, tmp_path, monkeypatch):
    """A `nexus.px4_sitl.ref` beside a project catalog overrides the shipped PX4 pin, and the receipt
    names that file.

    Given a project catalog with `nexus.px4_sitl.ref` naming a commit of another fork, when a PX4 run
    builds, then the run fetches that commit and flies it, and the receipt records it with the project file as
    its source.
    """
    sha = _fork(tmp_path, monkeypatch)
    project = _project(tmp_path, monkeypatch, **{"nexus.px4_sitl.ref": f"example/px4@{sha}\n"})

    outcome = _run(tmp_path)

    assert (_px4_tree(daemon), _pin(outcome, "px4_sitl")) == (
        str(tmp_path / "home" / ".cache" / "nexus" / "px4" / sha),
        {"ref": f"example/px4@{sha}", "source": str(project / "nexus.px4_sitl.ref"), "commit": None},
    ), outcome["error"]


def test_a_kit_run_with_no_project_pin_runs_the_shipped_image_and_records_it(daemon, tmp_path):
    """A Kit run with no project pin file and no variable runs the image of the Kit peer's shipped pin
    file, and the receipt records it.

    Given the stand-in daemon and a vehicle with a camera, when the run starts Kit, then the daemon runs
    `nvcr.io/nvidia/isaac-sim:6.0.1` and the receipt records it with the shipped file as its source.
    """
    outcome = _run(tmp_path, CAMERA, px4=False)

    assert ([r["image"] for r in daemon.started()], _pin(outcome, "kit")) == (
        ["nvcr.io/nvidia/isaac-sim:6.0.1"],
        {"ref": "nvcr.io/nvidia/isaac-sim:6.0.1", "source": str(_shipped("kit")), "commit": None},
    ), outcome["error"]


def test_a_projects_kit_pin_file_overrides_the_kit_image_and_the_receipt_names_it(daemon, tmp_path, monkeypatch):
    """A `nexus.kit.ref` beside a project catalog overrides the Kit image, and the receipt names that
    file.

    Given a project catalog with `nexus.kit.ref` naming another image this machine lacks, when the run
    starts Kit, then the daemon pulls and runs that image, and the receipt records it with the project
    file as its source.
    """
    daemon.held = set()
    project = _project(tmp_path, monkeypatch, **{"nexus.kit.ref": "registry.example/kit:7\n"})

    outcome = _run(tmp_path, CAMERA, px4=False)

    assert (daemon.pulls, [r["image"] for r in daemon.started()], _pin(outcome, "kit")) == (
        ["registry.example/kit:7"],
        ["registry.example/kit:7"],
        {"ref": "registry.example/kit:7", "source": str(project / "nexus.kit.ref"), "commit": None},
    ), outcome["error"]


def test_the_px4_variable_beats_a_projects_pin_file_and_the_receipt_names_it(daemon, tmp_path, monkeypatch):
    """`NEXUS_PX4_SITL_REF` set to a `repo@sha` beats a project `nexus.px4_sitl.ref`, and the receipt
    names the variable.

    Given a project `nexus.px4_sitl.ref` naming one commit and the variable naming another, when a PX4
    run builds, then the run fetches the variable's commit and flies it, and the receipt's source is the
    variable.
    """
    sha = _fork(tmp_path, monkeypatch)
    _project(tmp_path, monkeypatch, **{"nexus.px4_sitl.ref": "example/px4@" + "c" * 40 + "\n"})
    monkeypatch.setenv("NEXUS_PX4_SITL_REF", f"example/px4@{sha}")

    outcome = _run(tmp_path)

    assert (_px4_tree(daemon), _pin(outcome, "px4_sitl")) == (
        str(tmp_path / "home" / ".cache" / "nexus" / "px4" / sha),
        {"ref": f"example/px4@{sha}", "source": "NEXUS_PX4_SITL_REF", "commit": None},
    ), outcome["error"]


def test_the_px4_variable_naming_a_checkout_flies_it_and_records_its_head(daemon, tmp_path, monkeypatch):
    """`NEXUS_PX4_SITL_REF` set to a checkout's path flies that checkout with no fetch, and the receipt
    records the path, its HEAD commit and the variable.

    Given a stand-in checkout with one commit, when a PX4 run builds, then the peer flies that folder,
    and the receipt holds the path, the commit and the variable.
    """
    checkout = tmp_path / "my-px4"
    sha = _checkout(checkout)
    monkeypatch.setenv("NEXUS_PX4_SITL_REF", str(checkout))

    outcome = _run(tmp_path)

    assert (_px4_tree(daemon), _pin(outcome, "px4_sitl")) == (
        str(checkout),
        {"ref": str(checkout), "source": "NEXUS_PX4_SITL_REF", "commit": sha},
    ), outcome["error"]


def test_the_kit_variable_naming_a_local_image_runs_it_with_no_pull(daemon, tmp_path, monkeypatch):
    """`NEXUS_KIT_REF` set to a local image tag runs that image with no pull, and the receipt names the
    variable.

    Given the stand-in daemon holding a local tag and nothing else, when the run starts Kit, then it runs
    that tag, pulls nothing, and the receipt records the tag with the variable as its source.
    """
    daemon.held = {"nexus-kit:dev"}
    monkeypatch.setenv("NEXUS_KIT_REF", "nexus-kit:dev")

    outcome = _run(tmp_path, CAMERA, px4=False)

    assert (daemon.pulls, [r["image"] for r in daemon.started()], _pin(outcome, "kit")) == (
        [],
        ["nexus-kit:dev"],
        {"ref": "nexus-kit:dev", "source": "NEXUS_KIT_REF", "commit": None},
    ), outcome["error"]


def test_a_pin_file_that_names_no_peer_fails_the_run_before_any_peer_starts(daemon, tmp_path, monkeypatch):
    """A `nexus.<name>.ref` beside the catalog that names no peer fails the run before any peer starts,
    naming the file and the peer names it accepts.

    Given a project catalog with `nexus.px4.ref` beside it and a vehicle whose camera needs Kit, when a
    run builds, then it raises naming `nexus.px4.ref`, `px4_sitl` and `kit`, and no container starts.
    """
    _project(tmp_path, monkeypatch, **{"nexus.px4.ref": "example/px4@" + "a" * 40 + "\n"})

    outcome = _run(tmp_path, CAMERA, peers={"px4_sitl": Px4Fake})

    message = str(outcome["error"] or "")
    assert ([s in message for s in ("nexus.px4.ref", "px4_sitl", "kit")], daemon.runs) == ([True, True, True], [])


@pytest.mark.parametrize("text", ["example/px4\n", ""], ids=["no-at", "empty"])
def test_a_pin_not_in_its_peers_form_fails_the_run_naming_the_source_and_the_form(daemon, tmp_path, monkeypatch, text):
    """A pin file or variable not in its peer's form fails the run, naming the source and the form it
    expects.

    Given a `nexus.px4_sitl.ref` with no `@`, and separately an empty one, when a PX4 run builds, then it
    raises a `ValueError` naming the file and `repo@sha`.
    """
    project = _project(tmp_path, monkeypatch, **{"nexus.px4_sitl.ref": text})

    outcome = _run(tmp_path, peers={"px4_sitl": Px4Fake})

    error = outcome["error"]
    assert (
        isinstance(error, ValueError),
        str(project / "nexus.px4_sitl.ref") in str(error),
        "repo@sha" in str(error),
    ) == (
        True,
        True,
        True,
    ), error


def test_a_run_that_starts_no_kit_peer_records_no_kit_pin(daemon, tmp_path):
    """A run that starts no Kit peer records no Kit pin.

    Given a PX4 run on a vehicle with no camera or lidar, when it builds, then the receipt holds no Kit
    pin.
    """
    outcome = _run(tmp_path, peers={"px4_sitl": Px4Fake})

    assert (outcome["error"], _pin(outcome, "kit")) == (None, None)
