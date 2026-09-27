"""The PX4 Software In The Loop (SITL) peer: the container it starts, field by field, that the
runner clears a same-name leftover first, that a missing tree names its fix, that the container runs
as the tree's owner, that the peer's stop removes only what it started, and that this module stays
the only definition of the container, see GH #86.
"""

import os
import pathlib

import pytest
import yaml
from docker.errors import ImageNotFound, NotFound

import nexus._src.containers as containers
from nexus._src.vehicle.controllers.px4.sitl import IMAGE, Px4Sitl, build

ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture
def daemon(monkeypatch):
    """A stand-in docker daemon that already holds the px4-sitl image: it records what the peer asks
    of it, and touches nothing real.
    """
    calls = []

    class Images:
        def get(self, tag):
            if tag.startswith(f"{IMAGE}:"):
                return object()
            raise ImageNotFound(tag)

    class Containers:
        def run(self, image, **kwargs):
            calls.append(("run", {"image": image, **kwargs}))
            return b"" if not kwargs.get("detach", True) else object()

        def get(self, name):
            calls.append(("get", name))
            raise NotFound(name)  # no leftover in a test

    class Client:
        images = Images()
        containers = Containers()

    c = Client()
    monkeypatch.setattr(containers, "client", lambda: c)
    monkeypatch.setattr(containers, "_pump_logs", lambda container, log_path: None)
    return calls


@pytest.fixture
def tree(tmp_path):
    """A stand-in PX4 tree: a folder with a Makefile."""
    px4 = tmp_path / "px4"
    px4.mkdir()
    (px4 / "Makefile").write_text("px4_sitl:\n")
    return px4


def _peer(tree, tmp_path, **kw):
    args = {
        "tree": tree,
        "airframe": "astro_max",
        "instance": 0,
        "name": "px4-under-test",
        "log_path": str(tmp_path / "px4.log"),
    }
    return Px4Sitl(**{**args, **kw})


def test_start_runs_px4_with_the_runs_instance_and_airframe(daemon, tree, tmp_path):
    """The container definition is a contract, field by field. The peer runs PX4's binary itself,
    the way PX4's multi-instance script does, since ``make px4_sitl none_<airframe>`` can't take an
    instance; ``-d`` runs with no pxh shell, which at the end of an unattended stdin spin-loops
    printing its prompt and starves PX4.
    """
    _peer(tree, tmp_path, instance=2).start()

    launch = next(kw for kind, kw in daemon if kind == "run" and kw.get("detach", True))
    assert launch == {
        "image": launch["image"],
        "command": ["build/px4_sitl_default/bin/px4", "-i", "2", "-d"],
        "name": "px4-under-test",
        "detach": True,
        "user": f"{os.getuid()}:{os.getgid()}",
        "environment": {"HOME": "/tmp", "PX4_SIM_MODEL": "none_astro_max"},
        "working_dir": str(tree),
        "volumes": {str(tree): {"bind": str(tree), "mode": "rw"}},
        "auto_remove": True,
        "network_mode": "host",
    } and launch["image"].startswith(f"{IMAGE}:")


def test_start_builds_the_tree_before_it_launches(daemon, tree, tmp_path):
    """A launch must reach the sim inside its preroll window, which an incremental build fits and a
    cold build never does, so the build comes first, in the foreground.
    """
    _peer(tree, tmp_path).start()

    commands = [kw["command"] for kind, kw in daemon if kind == "run"]
    assert commands == [["make", "px4_sitl"], ["build/px4_sitl_default/bin/px4", "-i", "0", "-d"]]


def test_start_clears_a_stale_container_of_its_name(daemon, tree, tmp_path):
    """A leftover from a killed process blocks the name, so the runner clears it before creating the new one."""
    _peer(tree, tmp_path).start()

    kinds = [kind for kind, _ in daemon]
    assert kinds.index("get") < len(kinds) - 1 and kinds[-1] == "run"


def test_build_names_the_fix_when_the_tree_is_missing(monkeypatch, tmp_path):
    """Without the check, docker creates the empty mount and make reports "No rule to make target
    'px4_sitl'", which reads as a broken image rather than a missing tree.
    """
    missing = tmp_path / "nope"

    def never(*a, **kw):
        raise AssertionError("the daemon must not be touched when there is no tree")

    monkeypatch.setattr(containers, "client", never)

    with pytest.raises(RuntimeError, match=str(missing)):
        build(missing)


def test_container_user_is_the_trees_owner(daemon, tree, tmp_path, monkeypatch):
    """Inside the Kit container this process is root, so `os.getuid()` would build the bind-mounted
    host tree as uid 0 and leave its build/ folder root-owned. The tree's own owner is right in both places.
    """
    real_stat = pathlib.Path.stat

    class _Stat:
        st_uid, st_gid = 4242, 4343

    monkeypatch.setattr(pathlib.Path, "stat", lambda self, **kw: _Stat() if self == tree else real_stat(self, **kw))

    _peer(tree, tmp_path).start()

    assert {kw["user"] for kind, kw in daemon if kind == "run"} == {"4242:4343"}


def test_compose_does_not_redefine_the_container():
    """The container has one definition: this module. A second one lived in docker-compose.yml as
    the `px4-sitl` service and drifted from it for months unnoticed, with the wrong user, no home directory at /tmp,
    and a hardcoded telemetry port, because nothing executable ran it. Re-adding it starts that over.
    """
    doc = yaml.safe_load((ROOT / "docker" / "docker-compose.yml").read_text())
    assert "px4-sitl" not in doc["services"]


def test_stop_leaves_a_container_this_peer_never_started(daemon, tree, tmp_path):
    """Another run's container of the same name is that run's live flight."""
    _peer(tree, tmp_path).stop()

    assert daemon == [], "teardown touched the daemon without ever having started"


def test_stop_removes_the_container_this_peer_did_start(daemon, tree, tmp_path):
    peer = _peer(tree, tmp_path)
    peer.start()
    daemon.clear()

    peer.stop()

    assert ("get", "px4-under-test") in daemon  # the force-remove by name
