"""The PX4 Software In The Loop (SITL) peer: the container it starts, field by field, that the
runner clears a same-name leftover first, that a start removes a leftover of its instance whose
process has exited and fails while a live one holds it, that a missing tree names its fix, that the
container runs as the tree's owner, and that the peer's stop removes only what it started.
"""

import os
import pathlib
import subprocess

import pytest
from docker.errors import ImageNotFound, NotFound
from docker.types import Mount

import nexus_uav._src.peers.containers as containers
from nexus_uav._src.peers.pins import PeerPin
from nexus_uav._src.peers.px4_sitl.runner import IMAGE, Px4Sitl, build


class _Calls(list):
    """What the peer asked of the stand-in daemon, in order; ``held`` seeds the containers it runs already, and
    ``console`` is what a container the peer starts has printed.
    """

    def __init__(self):
        super().__init__()
        self.held: list[_Held] = []
        self.console = b""


class _Started:
    """A container the peer started on the stand-in daemon, which has printed its daemon's ``console``."""

    def __init__(self, calls):
        self._calls = calls

    def logs(self, **kwargs):
        return self._calls.console


class _Held:
    """A container the daemon runs before the test starts: a leftover, or another run's."""

    def __init__(self, calls, name, *, instance, owner):
        self._calls = calls
        self.name = name
        self.labels = {"nexus.peer": "px4_sitl", "nexus.px4.instance": str(instance), "nexus.owner": str(owner)}

    def remove(self, force=False):
        self._calls.append(("remove", self.name))


def _dead_pid() -> int:
    """The pid of a process that has exited."""
    process = subprocess.Popen(["true"])
    process.wait()
    return process.pid


@pytest.fixture
def daemon(monkeypatch, tmp_path):
    """A stand-in docker daemon that already holds the px4-sitl image: it records what the peer asks
    of it, and touches nothing real. Home is a test folder too, so a start empties no real instance's
    working folder, which a live run can hold.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    calls = _Calls()

    class Images:
        def get(self, tag):
            if tag.startswith(f"{IMAGE}:"):
                return object()
            raise ImageNotFound(tag)

    class Containers:
        def run(self, image, **kwargs):
            calls.append(("run", {"image": image, **kwargs}))
            return b"" if not kwargs.get("detach", True) else _Started(calls)

        def get(self, name):
            calls.append(("get", name))
            raise NotFound(name)  # no leftover of the same name in a test

        def list(self, **kwargs):
            wanted = dict(f.split("=", 1) for f in (kwargs.get("filters") or {}).get("label", []))
            return [c for c in calls.held if wanted.items() <= c.labels.items()]

    class Client:
        images = Images()
        containers = Containers()

    c = Client()
    monkeypatch.setattr(containers, "client", lambda: c)
    monkeypatch.setattr(containers, "_pump_logs", lambda container, log_path: None)
    return calls


@pytest.fixture
def tree(tmp_path):
    """A stand-in PX4 tree, a folder with a Makefile, that the peer's pin names so its start finds it."""
    px4 = tmp_path / "px4"
    px4.mkdir()
    (px4 / "Makefile").write_text("px4_sitl:\n")
    return px4


@pytest.fixture
def pty():
    """A pseudo-terminal of the test's own, as the run makes a serial link's end: the path of its slave, which this
    process owns, closed after the test.
    """
    master, slave = os.openpty()
    yield os.ttyname(slave)
    os.close(master)
    os.close(slave)


def _owned_by(monkeypatch, tree, uid: int, gid: int) -> None:
    """Make `tree` read as owned by `uid:gid`, as a tree another user built does, with the rest of its stat real."""
    real_stat = pathlib.Path.stat

    class _Stat:
        st_uid, st_gid = uid, gid

        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

    monkeypatch.setattr(
        pathlib.Path, "stat", lambda self, **kw: _Stat(real_stat(self, **kw)) if self == tree else real_stat(self, **kw)
    )


def _peer(tree, tmp_path, **kw):
    args = {
        "pin": PeerPin(ref=str(tree), source="a test"),
        "airframe": "astro_max",
        "instance": 0,
        "log_dir": tmp_path / "run",
    }
    return Px4Sitl(**{**args, **kw})


def test_start_runs_px4_with_the_runs_instance_and_airframe(daemon, tree, tmp_path):
    """The container definition is a contract, field by field. The peer runs PX4's binary itself,
    the way PX4's multi-instance script does, since ``make px4_sitl none_<airframe>`` can't take an
    instance; ``-d`` runs with no pxh shell, which at the end of an unattended stdin spin-loops
    printing its prompt and starves PX4. The instance's own working folder mounts over its folder of
    the tree, so PX4 boots with no state an earlier run left there, and the run's folder mounts over
    PX4's log root inside it, so PX4 writes the run's ULog there and leaves none in the tree. The container
    doesn't remove itself, so a PX4 that exits at once leaves its console for the log.
    """
    _peer(tree, tmp_path, instance=2).start()

    launch = next(kw for kind, kw in daemon if kind == "run" and kw.get("detach", True))
    assert launch == {
        "image": launch["image"],
        "command": ["build/px4_sitl_default/bin/px4", "-i", "2", "-d"],
        "name": f"nexus-px4_sitl-{os.getpid()}-2",
        "detach": True,
        "user": f"{os.getuid()}:{os.getgid()}",
        "environment": {"HOME": "/tmp", "PX4_SIM_MODEL": "none_astro_max"},
        "working_dir": str(tree),
        "mounts": [
            Mount(str(tree), str(tree), type="bind", read_only=False),
            Mount(
                str(tree / "build/px4_sitl_default/rootfs/2"),
                str(tmp_path / "home" / ".cache" / "nexus" / "px4-work" / "2"),
                type="bind",
                read_only=False,
            ),
            Mount(
                str(tree / "build/px4_sitl_default/rootfs/2/log"),
                str(tmp_path / "run" / "px4_sitl"),
                type="bind",
                read_only=False,
            ),
        ],
        "network_mode": "host",
        "labels": {"nexus.peer": "px4_sitl", "nexus.px4.instance": "2", "nexus.owner": str(os.getpid())},
    } and launch["image"].startswith(f"{IMAGE}:")


def test_start_mounts_each_serial_lines_device_and_starts_a_mavlink_instance_on_it_from_px4s_post_hook(
    daemon, tree, tmp_path, pty
):
    """A peer with a serial line mounts its end into the container as `/dev/serial/<prim>`, named for the prim that
    wires the line, and PX4 starts a MAVLink instance on it from its own `<airframe>.post` hook, a file of this run's
    mounted over the tree's copy.

    Given a peer whose end of the line from `Companion` is a pseudo-terminal of the test's, over a tree whose built
    startup folder holds the airframe `80000_none_astro_max`, when it starts, then the container binds that terminal
    at `/dev/serial/Companion` read-write, and the run's `serial.post` read-only at `80000_none_astro_max.post`
    beside the airframe, holding `mavlink start -d /dev/serial/Companion -b 3000000 -m onboard -x`.
    """
    airframes = tree / "build" / "px4_sitl_default" / "etc" / "init.d-posix" / "airframes"
    airframes.mkdir(parents=True)
    (airframes / "80000_none_astro_max").write_text("# the airframe\n")
    post = tmp_path / "home" / ".cache" / "nexus" / "px4-work" / "0" / "serial.post"

    _peer(tree, tmp_path, serial={"Companion": pty}).start()

    launch = next(kw for kind, kw in daemon if kind == "run" and kw.get("detach", True))
    added = [m for m in launch["mounts"] if m["Source"] in (pty, str(post))]
    expected = [
        Mount(f"{airframes}/80000_none_astro_max.post", str(post), type="bind", read_only=True),
        Mount("/dev/serial/Companion", pty, type="bind", read_only=False),
    ]
    assert (added, post.read_text()) == (expected, "mavlink start -d /dev/serial/Companion -b 3000000 -m onboard -x\n")


def test_start_gives_px4s_user_each_serial_end_this_process_owns(daemon, tree, tmp_path, pty, monkeypatch):
    """The run makes each serial end as this process, and PX4 runs as the tree's owner, so when the two differ PX4
    couldn't open its end: the start gives the end to PX4's user first.

    Given a tree owned by `4242:4343` and an end this process owns, when the peer starts, then it made `4242` the
    end's owner, its group untouched, and the container runs as `4242:4343`.
    """
    _owned_by(monkeypatch, tree, 4242, 4343)
    given = []
    monkeypatch.setattr(os, "chown", lambda path, uid, gid: given.append((path, uid, gid)))

    _peer(tree, tmp_path, serial={"Companion": pty}).start()

    users = [kw["user"] for kind, kw in daemon if kind == "run" and kw.get("detach", True)]
    assert (given, users) == ([(pty, 4242, -1)], ["4242:4343"])


def test_start_fails_naming_the_serial_end_and_the_user_when_it_cannot_give_it(
    daemon, tree, tmp_path, pty, monkeypatch
):
    """A process that can't give an end away fails the start, before PX4 runs, naming the device and the user.

    Given a tree owned by `4242:4343`, an end this process owns and a kernel that refuses the change of owner, when
    the peer starts, then it fails naming the end's path and `4242`, and the daemon ran the build alone.
    """
    _owned_by(monkeypatch, tree, 4242, 4343)

    def refuse(path, uid, gid):
        raise PermissionError(1, "Operation not permitted", path)

    monkeypatch.setattr(os, "chown", refuse)

    with pytest.raises(RuntimeError) as e:
        _peer(tree, tmp_path, serial={"Companion": pty}).start()

    commands = [kw["command"] for kind, kw in daemon if kind == "run"]
    assert (pty in str(e.value), "4242" in str(e.value), commands) == (True, True, [["make", "px4_sitl"]]), str(e.value)


def test_start_empties_the_instances_work_folder_of_what_an_earlier_run_left(daemon, tree, tmp_path):
    """A start empties the instance's work folder, so PX4 boots with no parameter and no mission an earlier run
    stored there, whether that run ended or died before its teardown: the peer finds one folder either way.

    Given the instance's work folder holding `parameters.bson` and `dataman`, when the peer starts, then the folder
    holds only the empty `log` the run's folder mounts over.
    """
    work = tmp_path / "home" / ".cache" / "nexus" / "px4-work" / "0"
    work.mkdir(parents=True)
    (work / "parameters.bson").write_bytes(b"left by an earlier run")
    (work / "dataman").write_bytes(b"left by an earlier run")

    _peer(tree, tmp_path).start()

    assert sorted(p.name for p in work.iterdir()) == ["log"]


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


def test_start_removes_a_leftover_of_its_instance_whose_process_has_exited(daemon, tree, tmp_path):
    """PX4 SITL outlives a run killed without its teardown and keeps dialing the sim's port, and no
    later run shares its container's name, so the next start on that instance removes it first.
    """
    daemon.held.append(_Held(daemon, "nexus-px4_sitl-1-0", instance=0, owner=_dead_pid()))

    _peer(tree, tmp_path).start()

    kinds = [kind for kind, _ in daemon]
    assert (kinds[0], kinds[-1]) == ("remove", "run")


def test_start_fails_naming_the_live_process_that_holds_its_instance(daemon, tree, tmp_path):
    """Two live PX4s on one instance would share every port, so the start fails and names the holder,
    and neither removes the live run's autopilot nor starts a second one.
    """
    holder = os.getppid()
    daemon.held.append(_Held(daemon, "nexus-px4_sitl-7-0", instance=0, owner=holder))

    with pytest.raises(RuntimeError, match=f"instance 0 is in use by process {holder}"):
        _peer(tree, tmp_path).start()

    assert list(daemon) == []


def test_start_leaves_a_container_of_another_instance_alone(daemon, tree, tmp_path):
    """Runs on two instances fly at once, so a start touches only its own instance's containers."""
    daemon.held.append(_Held(daemon, "nexus-px4_sitl-7-1", instance=1, owner=os.getppid()))

    _peer(tree, tmp_path).start()

    assert [kind for kind, _ in daemon if kind in ("remove", "run")] == ["run", "run"]


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
    _owned_by(monkeypatch, tree, 4242, 4343)

    _peer(tree, tmp_path).start()

    assert {kw["user"] for kind, kw in daemon if kind == "run"} == {"4242:4343"}


def test_stop_leaves_a_container_this_peer_never_started(daemon, tree, tmp_path):
    """Another run's container of the same name is that run's live flight."""
    _peer(tree, tmp_path).stop()

    assert daemon == [], "teardown touched the daemon without ever having started"


def test_stop_removes_the_container_this_peer_did_start(daemon, tree, tmp_path):
    peer = _peer(tree, tmp_path)
    peer.start()
    daemon.clear()

    peer.stop()

    assert ("get", f"nexus-px4_sitl-{os.getpid()}-0") in daemon  # the force-remove by name, its process and instance


def test_px4s_error_lines_come_whole_from_its_console_before_the_log_holds_them(daemon, tree, tmp_path):
    """PX4's error lines come whole from its container's console, before its log file holds them, #321.

    A PX4 that exits at once can fail the run before the log pump has written its console to `console.log`, and the
    run's error must still quote PX4's own line. Given a started PX4 whose container has printed its instance and
    then the unknown model line, and whose log file is still empty, when the run reads PX4's error lines, then it
    reads the unknown model line alone.
    """
    daemon.console = b"INFO  [px4] instance: 0\nERROR [init] Unknown model none_astro_max\n"
    peer = _peer(tree, tmp_path)
    peer.start()

    assert peer.errors() == ["ERROR [init] Unknown model none_astro_max"]
