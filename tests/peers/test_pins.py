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


def _run(
    tmp_path, body: str = "", *, px4: bool = True, peers: dict | None = None, root: str = "", registry=None
) -> dict:
    """Build the fixture vehicle, with ``body`` under its base body, ``root`` under its root prim and with the PX4
    Software In The Loop (SITL) peer declared when ``px4``, in the fixture scene, with ``registry`` when given.
    Returns what the run left: its receipt as JSON, or the error the build raised, which ends a run as well.
    """
    launch = LaunchConfig.from_dict(
        {"vehicle": sv.vehicle(tmp_path, body, root=root, px4=px4), "scene": sv.SCENE, "runtime": {"device": "cpu"}}
    )
    by_registry = {} if registry is None else {"components": registry}
    try:
        loop = L.build_from_launch(launch, preroll_timeout=10.0, peers=peers, **by_registry)
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


@pytest.mark.parametrize(
    ("source", "text"),
    [("file", "example/px4\n"), ("file", ""), ("variable", "")],
    ids=["no-at", "empty", "empty-variable"],
)
def test_a_pin_not_in_its_peers_form_fails_the_run_naming_the_source_and_the_form(
    daemon, tmp_path, monkeypatch, source, text
):
    """A pin file or variable not in its peer's form fails the run, naming the source and the form it
    expects.

    Given a `nexus.px4_sitl.ref` with no `@`, separately an empty one, and separately an empty
    `NEXUS_PX4_SITL_REF`, when a PX4 run builds, then it raises a `ValueError` naming the file or the
    variable, and `repo@sha`.
    """
    if source == "file":
        named = str(_project(tmp_path, monkeypatch, **{"nexus.px4_sitl.ref": text}) / "nexus.px4_sitl.ref")
    else:
        monkeypatch.setenv("NEXUS_PX4_SITL_REF", text)
        named = "NEXUS_PX4_SITL_REF"

    outcome = _run(tmp_path, peers={"px4_sitl": Px4Fake})

    error = outcome["error"]
    assert (isinstance(error, ValueError), named in str(error), "repo@sha" in str(error)) == (True, True, True), error


def test_a_run_that_starts_no_kit_peer_records_no_kit_pin(daemon, tmp_path):
    """A run that starts no Kit peer records no Kit pin.

    Given a PX4 run on a vehicle with no camera or lidar, when it builds, then the receipt holds no Kit
    pin.
    """
    outcome = _run(tmp_path, peers={"px4_sitl": Px4Fake})

    assert (outcome["error"], _pin(outcome, "kit")) == (None, None)


def test_a_px4_pin_naming_a_path_with_no_folder_fails_the_run_before_any_peer_starts(daemon, tmp_path, monkeypatch):
    """A PX4 pin naming a path with no folder behind it fails the run before any peer starts.

    Given `NEXUS_PX4_SITL_REF` set to a path where no folder exists, when a PX4 run builds, then it
    raises a `ValueError` naming the variable and the path, and no container starts.
    """
    missing = tmp_path / "no-such-px4"
    monkeypatch.setenv("NEXUS_PX4_SITL_REF", str(missing))

    outcome = _run(tmp_path)

    error = outcome["error"]
    named = [s in str(error) for s in ("NEXUS_PX4_SITL_REF", str(missing))]
    assert (isinstance(error, ValueError), named, daemon.runs) == (True, [True, True], []), error


EMPTY_CATALOG = "vehicles: {}\nscenes: {}\n"  # a package's catalog that adds no entry, beside its pin files


def test_an_installed_packages_kit_pin_file_is_the_pin_the_kit_peer_runs(daemon, tmp_path, install_package):
    """A package's pin file is the pin a peer flies when no project pin file or variable sets it.

    Given a package shipping `nexus.kit.ref` naming an image this machine lacks, when a run with no project
    starts Kit, then the daemon pulls and runs that image, and the receipt records it with the package's file
    as its source.
    """
    daemon.held = set()
    package = install_package(
        "acme_drones", {"nexus.catalog.yaml": EMPTY_CATALOG, "nexus.kit.ref": "registry.example/kit:8\n"}
    )

    outcome = _run(tmp_path, CAMERA, px4=False)

    assert (daemon.pulls, [r["image"] for r in daemon.started()], _pin(outcome, "kit")) == (
        ["registry.example/kit:8"],
        ["registry.example/kit:8"],
        {"ref": "registry.example/kit:8", "source": str(package / "nexus.kit.ref"), "commit": None},
    ), outcome["error"]


def test_a_projects_kit_pin_file_overrides_an_installed_packages(daemon, tmp_path, monkeypatch, install_package):
    """A project's pin file overrides a package's pin file for the same peer.

    Given a package and a project each shipping `nexus.kit.ref`, when a run in the project starts Kit, then
    the daemon runs the project's image, and the receipt names the project's file.
    """
    install_package("acme_drones", {"nexus.catalog.yaml": EMPTY_CATALOG, "nexus.kit.ref": "registry.example/kit:8\n"})
    project = _project(tmp_path, monkeypatch, **{"nexus.kit.ref": "registry.example/kit:7\n"})

    outcome = _run(tmp_path, CAMERA, px4=False)

    assert ([r["image"] for r in daemon.started()], _pin(outcome, "kit")) == (
        ["registry.example/kit:7"],
        {"ref": "registry.example/kit:7", "source": str(project / "nexus.kit.ref"), "commit": None},
    ), outcome["error"]


def test_the_kit_variable_overrides_an_installed_packages_pin_file(daemon, tmp_path, monkeypatch, install_package):
    """The peer's variable overrides a package's pin file.

    Given a package shipping `nexus.kit.ref` and `NEXUS_KIT_REF` set, when a run starts Kit, then the daemon
    runs the variable's image, and the receipt names the variable.
    """
    install_package("acme_drones", {"nexus.catalog.yaml": EMPTY_CATALOG, "nexus.kit.ref": "registry.example/kit:8\n"})
    monkeypatch.setenv("NEXUS_KIT_REF", "nexus-kit:dev")

    outcome = _run(tmp_path, CAMERA, px4=False)

    assert ([r["image"] for r in daemon.started()], _pin(outcome, "kit")) == (
        ["nexus-kit:dev"],
        {"ref": "nexus-kit:dev", "source": "NEXUS_KIT_REF", "commit": None},
    ), outcome["error"]


def test_two_installed_packages_that_pin_one_peer_fail_the_run_naming_it_and_both(daemon, tmp_path, install_package):
    """Two installed packages that ship a pin file for one peer fail the run, and the message names the peer
    and both packages.

    Given two packages each shipping `nexus.px4_sitl.ref` and no project file or variable, when a PX4 run
    builds, then it fails and the message names `px4_sitl` and both packages.
    """
    for name in ("acme_drones", "bravo_drones"):
        install_package(
            name, {"nexus.catalog.yaml": EMPTY_CATALOG, "nexus.px4_sitl.ref": "example/px4@" + "a" * 40 + "\n"}
        )

    outcome = _run(tmp_path, peers={"px4_sitl": Px4Fake})

    message = str(outcome["error"] or "")
    assert [s in message for s in ("px4_sitl", "acme_drones", "bravo_drones")] == [True, True, True], message


def test_an_installed_packages_pin_file_that_names_no_peer_fails_the_run(daemon, tmp_path, install_package):
    """A package's pin file that names no peer fails the run, and the message names the file.

    Given a package shipping `nexus.nosuch.ref`, when a run builds, then it fails naming that file and the
    accepted pin file names, and no container starts.
    """
    package = install_package("acme_drones", {"nexus.catalog.yaml": EMPTY_CATALOG, "nexus.nosuch.ref": "x\n"})

    outcome = _run(tmp_path, CAMERA, peers={"px4_sitl": Px4Fake})

    message = str(outcome["error"] or "")
    named = [s in message for s in (str(package / "nexus.nosuch.ref"), "nexus.px4_sitl.ref", "nexus.kit.ref")]
    assert (named, daemon.runs) == ([True, True, True], [])


def test_a_peer_a_project_declares_flies_the_pin_its_package_ships(daemon, tmp_path, install_package):
    """A peer a project declares flies the pin its package ships.

    Given an installed package whose catalog layer holds `nexus.radio.ref` for its own peer `radio`, and the
    fixture vehicle that declares that peer, when a run builds, then the peer receives that pin and the receipt
    records it with the package's file as its source.
    """
    package = install_package("acme_radio", {"nexus.catalog.yaml": EMPTY_CATALOG, "nexus.radio.ref": "radio@1.2\n"})
    peer = sv.radio()
    registry = sv.components(NexusPx4SitlAPI=Px4Fake, StandInPeerAPI=peer)

    outcome = _run(tmp_path, root=sv.companion(), registry=registry)

    pin = {"ref": "radio@1.2", "source": str(package / "nexus.radio.ref"), "commit": None}
    given = peer.built[0].get("pin") if peer.built else None
    given = given.model_dump() if hasattr(given, "model_dump") else given
    assert (given, _pin(outcome, "radio")) == (pin, pin), outcome["error"]


def test_a_projects_peer_pin_not_in_the_form_its_peer_states_fails_before_any_peer_starts(
    daemon, tmp_path, install_package
):
    """A project's peer pin not in the form its peer states fails the run before any peer starts.

    Given an installed package whose `nexus.radio.ref` reads `radio`, not the `name@version` its peer states,
    and the fixture vehicle that declares that peer, when a run builds, then it raises naming the pin file and
    the form, and no peer started.
    """
    package = install_package("acme_radio", {"nexus.catalog.yaml": EMPTY_CATALOG, "nexus.radio.ref": "radio\n"})
    peer = sv.radio()
    registry = sv.components(NexusPx4SitlAPI=Px4Fake, StandInPeerAPI=peer)

    outcome = _run(tmp_path, root=sv.companion(), registry=registry)

    message = str(outcome["error"] or "")
    named = [s in message for s in (str(package / "nexus.radio.ref"), "name@version")]
    assert (named, peer.events, daemon.runs) == ([True, True], [], []), outcome["error"]
