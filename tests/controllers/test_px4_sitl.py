"""The PX4 Software In The Loop (SITL) launcher: the container kwargs ``launch()`` builds, that it clears
a stale same-name container first, that the env resolves per call, that a missing checkout names its fix,
that the container runs as the checkout's owner, and that this module stays the only definition of the
container, see GH #86.
"""

import os
import pathlib

import pytest
import yaml

from nexus._src.vehicle.controllers.px4.sitl import Px4Sitl, _container_kwargs, build_px4_sitl

ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture
def daemon(monkeypatch):
    """A fake docker daemon: records what the launcher asks of it, and touches nothing real."""
    import docker.errors

    import nexus._src.containers as containers

    calls = []

    class FakeContainers:
        def run(self, image, **kwargs):
            calls.append(("run", {"image": image, **kwargs}))
            return object()

        def get(self, name):
            calls.append(("get", name))
            raise docker.errors.NotFound(name)  # no leftover in a test

    class FakeClient:
        containers = FakeContainers()

    monkeypatch.setattr(containers, "client", lambda: FakeClient())
    monkeypatch.setattr(containers, "_pump_logs", lambda container, log_path: None)
    return calls


def test_env_resolves_per_call(monkeypatch, tmp_path):
    """$PX4_DIR / $PX4_IMAGE set *after* import still take effect: the launcher reads them per call, and
    doesn't capture them in module-level constants, which is what it used to do.
    """
    monkeypatch.setenv("PX4_DIR", str(tmp_path))
    monkeypatch.setenv("PX4_IMAGE", "example.invalid/px4:pinned")
    kwargs = _container_kwargs()
    assert kwargs["working_dir"] == str(tmp_path)
    assert kwargs["volumes"] == {str(tmp_path): {"bind": str(tmp_path), "mode": "rw"}}
    assert kwargs["image"] == "example.invalid/px4:pinned"


def test_launch_builds_the_expected_container_kwargs(daemon, monkeypatch, tmp_path):
    """The container definition is a contract, field by field. stdin must stay open: at the end of stdin
    the pxh shell spin-loops printing its prompt and starves PX4.
    """
    px4_dir = tmp_path / "px4"
    px4_dir.mkdir()
    monkeypatch.setenv("PX4_DIR", str(px4_dir))
    monkeypatch.setenv("PX4_IMAGE", "example.invalid/px4:pinned")

    log = tmp_path / "px4.log"
    Px4Sitl(container="px4-under-test", log_path=str(log), airframe="none_astro_max").launch()

    (_, kwargs) = next(c for c in daemon if c[0] == "run")
    assert kwargs == {
        "image": "example.invalid/px4:pinned",
        "command": ["make", "px4_sitl", "none_astro_max"],
        "name": "px4-under-test",
        "detach": True,
        "user": f"{os.getuid()}:{os.getgid()}",
        "environment": {"HOME": "/tmp"},
        "working_dir": str(px4_dir),
        "volumes": {str(px4_dir): {"bind": str(px4_dir), "mode": "rw"}},
        "auto_remove": True,
        "network_mode": "host",
        "stdin_open": True,
    }


def test_launch_clears_a_stale_container(daemon, monkeypatch, tmp_path):
    """A leftover from a killed process blocks the name and squats the Ground Control Station (GCS) ports, so the
    launcher clears it *before* creating the new one: the "one PX4 at a time" gotcha, retired in code.
    """
    px4_dir = tmp_path / "px4"
    px4_dir.mkdir()
    monkeypatch.setenv("PX4_DIR", str(px4_dir))

    Px4Sitl(container="px4-under-test", log_path=str(tmp_path / "px4.log")).launch()

    kinds = [c[0] for c in daemon]
    assert kinds.index("get") < kinds.index("run")  # force-remove by name, then create


def test_build_names_the_fix_when_the_checkout_is_missing(monkeypatch, tmp_path):
    """Without the check, docker creates the empty mount and make reports "No rule to make target
    'px4_sitl'" and that reads as a broken image rather than a missing checkout.
    """
    missing = tmp_path / "nope"
    monkeypatch.setenv("PX4_DIR", str(missing))

    import nexus._src.containers as containers

    def never(*a, **kw):
        raise AssertionError("the daemon must not be touched when there is no checkout")

    monkeypatch.setattr(containers, "client", never)

    with pytest.raises(RuntimeError, match=str(missing)):
        build_px4_sitl()


def test_container_user_is_the_checkout_owner(monkeypatch, tmp_path):
    """Inside the Kit container this process is root, so `os.getuid()` would build the bind-mounted
    host checkout as uid 0 and leave its build/ tree root-owned. The checkout's own owner is right
    in both places.
    """
    px4_dir = tmp_path / "px4"
    px4_dir.mkdir()
    monkeypatch.setenv("PX4_DIR", str(px4_dir))
    real_stat = pathlib.Path.stat

    class _Stat:
        st_uid, st_gid = 4242, 4343

    monkeypatch.setattr(pathlib.Path, "stat", lambda self, **kw: _Stat() if self == px4_dir else real_stat(self, **kw))
    assert _container_kwargs()["user"] == "4242:4343"


def test_compose_does_not_redefine_the_container():
    """The container has one definition: this module. A second one lived in docker-compose.yml as
    the `px4-sitl` service and drifted from it for months unnoticed, with the wrong user, no home directory at /tmp,
    and a hardcoded telemetry port, because nothing executable ran it. Re-adding it starts that over.
    """
    doc = yaml.safe_load((ROOT / "docker" / "docker-compose.yml").read_text())
    assert "px4-sitl" not in doc["services"]


def test_stop_leaves_a_container_this_run_never_launched(daemon, monkeypatch, tmp_path):
    """The name is a singleton, so a run that never launched must not remove it: that container is
    somebody else's live flight. A second sim on one host dies at its own :4560 bind, and its
    teardown used to take the healthy run's PX4 down with it.
    """
    px4_dir = tmp_path / "px4"
    px4_dir.mkdir()
    monkeypatch.setenv("PX4_DIR", str(px4_dir))

    Px4Sitl(container="px4-under-test", log_path=str(tmp_path / "px4.log")).stop()

    assert daemon == [], "teardown touched the daemon without ever having launched"


def test_stop_removes_the_container_this_run_did_launch(daemon, monkeypatch, tmp_path):
    px4_dir = tmp_path / "px4"
    px4_dir.mkdir()
    monkeypatch.setenv("PX4_DIR", str(px4_dir))

    sitl = Px4Sitl(container="px4-under-test", log_path=str(tmp_path / "px4.log"))
    sitl.launch()
    daemon.clear()
    sitl.stop()

    assert ("get", "px4-under-test") in daemon  # the force-remove by name
