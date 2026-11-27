"""The replay of a flight's ULog into PX4 Software In The Loop (SITL), #429: the rows it sends, the hold across a
gap in the stick rows, the PX4 it flies, the topics of an instance, the parameters PX4 boots on, and a whole
replayed flight.

The unit tests read a ULog the test writes, `ulog_file`. The test marked ``px4_sitl`` flies through ``nx.Sim``,
which builds and launches PX4 from the tree on this machine, so it needs docker and a PX4 tree carrying the
airframe the sim flies; it skips without one.
"""

import subprocess

import pytest

import nexus_uav._src.peers.containers as containers
from nexus_uav._src.peers.pins import PeerPin
from nexus_uav._src.peers.px4_sitl import checkout
from nexus_uav._src.peers.px4_sitl.replay import Sample, address, boot_params, px4_pin, samples
from nexus_uav.px4 import replay
from tests.peers.px4_sitl import ulog_file

_STICK = [
    ("uint64_t", "timestamp"),
    ("uint64_t", "timestamp_sample"),
    ("float", "roll"),
    ("float", "pitch"),
    ("float", "yaw"),
    ("float", "throttle"),
    ("bool", "valid"),
    ("uint8_t", "data_source"),
]
_COMMAND = [
    ("uint64_t", "timestamp"),
    ("float", "param1"),
    ("uint32_t", "command"),
    ("uint8_t", "target_system"),
    ("uint8_t", "source_system"),
    ("bool", "from_external"),
]
_LINK = [("uint64_t", "timestamp"), ("uint8_t", "type"), ("bool", "heartbeat_type_gcs")]


def _stick(t: int, roll: float, valid: bool = True) -> tuple:
    """A stick row at `t` microseconds with `roll`, the other axes at rest, from a MAVLink source."""
    return (t, t, roll, 0.0, 0.0, 0.5, valid, 2)


def _stick_values(t: int, roll: float) -> dict:
    """The fields of `_stick(t, roll)` by name."""
    return {
        "timestamp": t,
        "timestamp_sample": t,
        "roll": roll,
        "pitch": 0.0,
        "yaw": 0.0,
        "throttle": 0.5,
        "valid": True,
        "data_source": 2,
    }


def test_a_replay_sends_each_valid_stick_row_external_command_and_link_status_row_and_nothing_else(tmp_path):
    """A replay sends each valid stick row, each external command and each link status row of the log, and nothing
    else; a topic the log lacks sends nothing.

    Given a ULog with a valid and an invalid stick row, an external and an internal command, and a link status row,
    then the samples hold the valid row, the external command and the status row, each with its logged content and
    time. Given a ULog with no stick topic, then the samples hold the command and the status row alone.
    """
    commands = (_COMMAND, [(3_000_000, 1.0, 176, 1, 255, True), (3_500_000, 0.0, 512, 1, 1, False)])
    links = (_LINK, [(4_000_000, 3, True)])
    sticks = (_STICK, [_stick(1_000_000, 0.25), _stick(2_000_000, 0.5, valid=False)])
    whole = ulog_file.write(
        tmp_path / "whole.ulg",
        topics={"manual_control_setpoint": sticks, "vehicle_command": commands, "telemetry_status": links},
    )
    stickless = ulog_file.write(
        tmp_path / "stickless.ulg", topics={"vehicle_command": commands, "telemetry_status": links}
    )

    command = Sample(
        3_000_000,
        "vehicle_command",
        {
            "timestamp": 3_000_000,
            "param1": 1.0,
            "command": 176,
            "target_system": 1,
            "source_system": 255,
            "from_external": True,
        },
    )
    link = Sample(4_000_000, "telemetry_status", {"timestamp": 4_000_000, "type": 3, "heartbeat_type_gcs": True})
    stick = Sample(1_000_000, "manual_control_input", _stick_values(1_000_000, 0.25))
    assert (samples(whole), samples(stickless)) == ([stick, command, link], [command, link])


def test_a_replay_holds_the_last_stick_sample_across_a_gap_while_the_next_row_is_valid(tmp_path):
    """Across a gap in the stick rows, the replay holds the last sample while the next row is valid, and sends
    nothing more when the next row is invalid.

    Given stick rows at 1.0, 1.2 and 1.4 s and the next at 2.4 s, valid, then copies of the row at 1.4 s fill the
    gap at the rows' own period, 1.6 to 2.2 s, each with its own time. Given the row at 2.4 s invalid, then no
    copy.
    """
    before = [_stick(1_000_000, 0.0), _stick(1_200_000, 0.0), _stick(1_400_000, 0.25)]
    held = ulog_file.write(
        tmp_path / "held.ulg", topics={"manual_control_setpoint": (_STICK, [*before, _stick(2_400_000, 0.5)])}
    )
    lost = ulog_file.write(
        tmp_path / "lost.ulg",
        topics={"manual_control_setpoint": (_STICK, [*before, _stick(2_400_000, 0.5, valid=False)])},
    )

    def sent(ulog):
        return [(s.time, s.values["timestamp"], s.values["roll"]) for s in samples(ulog)]

    rows = [(1_000_000, 1_000_000, 0.0), (1_200_000, 1_200_000, 0.0), (1_400_000, 1_400_000, 0.25)]
    copies = [(t, t, 0.25) for t in (1_600_000, 1_800_000, 2_000_000, 2_200_000)]
    assert (sent(held), sent(lost)) == ([*rows, *copies, (2_400_000, 2_400_000, 0.5)], rows)


def test_a_replay_flies_the_px4_commit_the_log_names_or_the_px4_the_caller_names(tmp_path):
    """The replay flies the PX4 commit the log names, or the PX4 the caller names.

    Given a ULog whose `ver_sw` is commit `b…b`, a vehicle pinned to `PX4/PX4-Autopilot@a…a` and no PX4 named, then
    the pin is `PX4/PX4-Autopilot@b…b`. Given the PX4 `example/px4@c…c` named, then the pin is that one.
    """
    ulog = ulog_file.write(tmp_path / "flight.ulg", info={"ver_sw": "b" * 40})
    default = PeerPin(ref=f"PX4/PX4-Autopilot@{'a' * 40}", source="a test")

    pins = (px4_pin(ulog, None, default).ref, px4_pin(ulog, f"example/px4@{'c' * 40}", default).ref)

    assert pins == (f"PX4/PX4-Autopilot@{'b' * 40}", f"example/px4@{'c' * 40}")


@pytest.mark.usefixtures("warp_cpu")
def test_a_log_whose_commit_the_pins_repo_lacks_fails_before_any_peer_starts(tmp_path, monkeypatch):
    """A log whose commit the pin's repo lacks fails before any peer starts.

    Given a ULog whose `ver_sw` no fetch finds, when a script replays it on the fixture vehicle, then an error names
    the commit and says how to name a tree, the keyword `px4`, and no container started.
    """
    pytest.importorskip("newton")
    pytest.importorskip("pxr")
    from tests.usd import sensor_vehicle as sv

    commit = "0123456789abcdef0123456789abcdef01234567"
    ulog = ulog_file.write(tmp_path / "flight.ulg", info={"ver_sw": commit})
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # a cache that holds no PX4 tree
    monkeypatch.delenv("NEXUS_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("NEXUS_PX4_SITL_REF", raising=False)
    run = subprocess.run

    def no_such_commit(args, *a, **kw):
        """The network's answer to a fetch of a commit no repository holds; every other command runs."""
        if args[0] == "git" and "fetch" in args:
            raise subprocess.CalledProcessError(128, args, stderr="fatal: remote error: not our ref")
        return run(args, *a, **kw)

    monkeypatch.setattr(subprocess, "run", no_such_commit)
    daemon_calls = []

    def daemon():
        daemon_calls.append("client")
        raise RuntimeError("the test's docker daemon starts no container")

    monkeypatch.setattr(containers, "client", daemon)

    error = None
    try:
        replay(ulog, sv.vehicle(tmp_path, px4=True), scene=sv.SCENE, device="cpu", run_dir=tmp_path / "run")
    except Exception as exc:  # the test reads which error the replay raised
        error = exc

    assert (commit in str(error), "px4" in str(error), daemon_calls) == (True, True, []), error


def test_a_sample_for_another_instance_goes_to_its_topics_and_a_command_targets_its_system_id():
    """On an instance other than 0, each sample goes to that instance's topics and each command targets its system
    id.

    Given a mode command whose logged target is system 1, when it goes to instance 2, then its topic is
    `px4_2/fmu/in/vehicle_command` and its `target_system` is 3.
    """
    command = Sample(3_000_000, "vehicle_command", {"timestamp": 3_000_000, "command": 176, "target_system": 1})

    out = address(command, 2)

    assert (out.topic.removeprefix("rt/"), out.values["target_system"]) == ("px4_2/fmu/in/vehicle_command", 3)


def test_px4_boots_on_the_logs_parameters_without_the_skipped_ones_with_the_callers_delta_on_top(tmp_path):
    """PX4 boots on the log's parameters, minus the skipped ones, plus the caller's delta, with `UXRCE_DDS_SYNCT`
    0; the delta wins over the log.

    Given a ULog with `MPC_XY_VEL_MAX` 12, `MPC_Z_VEL_MAX_UP` 3, `UXRCE_DDS_SYNCT` 1, `COM_POWER_COUNT` 2,
    which denies arming on SITL's one power source, and `ZENOH_ENABLE` 1, which starts Zenoh in place of the bridge, and a delta of `MPC_XY_VEL_MAX` 5, then the boot parameters
    are `MPC_XY_VEL_MAX` 5, `MPC_Z_VEL_MAX_UP` 3 and `UXRCE_DDS_SYNCT` 0.
    """
    ulog = ulog_file.write(
        tmp_path / "flight.ulg",
        params={
            "MPC_XY_VEL_MAX": 12.0,
            "MPC_Z_VEL_MAX_UP": 3.0,
            "UXRCE_DDS_SYNCT": 1,
            "COM_POWER_COUNT": 2,
            "ZENOH_ENABLE": 1,
        },
    )

    booted = boot_params(ulog, {"MPC_XY_VEL_MAX": 5.0})

    assert booted == {"MPC_XY_VEL_MAX": 5.0, "MPC_Z_VEL_MAX_UP": 3.0, "UXRCE_DDS_SYNCT": 0}


# The tree the run would fly, with no fetch: the checkout the pin names, or the pinned tree once a run fetched it.
_TREE = checkout.tree(fetch_missing=False)
# A skip mark, not a skip at import, so a run that deselects `px4_sitl` lists no such test.
_NEEDS_TREE = pytest.mark.skipif(
    _TREE is None
    or not list((_TREE / "ROMFS" / "px4fmu_common" / "init.d-posix" / "airframes").glob("*_none_astro_max")),
    reason="needs docker and a PX4 tree on this machine, the pinned checkout or commit, carrying the none_astro_max airframe",
)


def _stand_in_dds(monkeypatch, *, matched: tuple[str, ...], tick: float, ends: float = float("inf")):
    """Stand in for the Data Distribution Service and the run: a writer whose topic ends in a name of `matched`
    has a reader, and the run's step advances `tick` seconds and answers false once the run reaches `ends`. Return the
    topics written, in order, and the run.
    """
    import cyclonedds.domain
    import cyclonedds.pub
    import cyclonedds.topic

    from nexus_uav._src.peers.px4_sitl import OFFBOARD_PORT
    from nexus_uav._src.peers.px4_sitl import replay as module

    written = []

    class Topic:
        def __init__(self, participant, name, kind):
            self.name = name

    class Writer:
        def __init__(self, participant, topic):
            self.topic = topic.name

        def get_matched_subscriptions(self):
            return [object()] if self.topic.endswith(matched) else []

        def write(self, message):
            written.append(self.topic)

    class Run:
        def __init__(self):
            self.time = 0.0
            self.ports = {"offboard": {"port": OFFBOARD_PORT}}

        def step(self):
            self.time += tick
            return self.time < ends

    monkeypatch.setattr(cyclonedds.domain, "DomainParticipant", lambda domain: object())
    monkeypatch.setattr(cyclonedds.pub, "DataWriter", Writer)
    monkeypatch.setattr(cyclonedds.topic, "Topic", Topic)
    monkeypatch.setattr(module, "_message_type", lambda tree, name: name)
    monkeypatch.setattr(module, "versions", lambda tree: dict.fromkeys(module._MESSAGES, 0))
    monkeypatch.setattr(module, "_message", lambda kind, values: values)
    monkeypatch.setattr(module, "_MATCH_HOLD_S", 0.0)
    return written, Run()


def test_a_replay_fails_when_a_due_sample_never_reaches_px4(tmp_path, monkeypatch):
    """A replay whose later topic no reader ever matches fails, and names the topic and the unsent samples.

    Given a ULog with a stick row at 1 s, then a command at 2 s and a link status row at 3 s, and a bridge that
    reads the sticks alone, when the replay flies to the log's end, then it raises a run error that names the
    command's topic and 2 unsent samples of 3, and the stick row alone went out.
    """
    from nexus_uav import RunError
    from nexus_uav._src.peers.px4_sitl import replay as module

    written, run = _stand_in_dds(monkeypatch, matched=("manual_control_input",), tick=0.5)
    log = module.read(
        ulog_file.write(
            tmp_path / "flight.ulg",
            topics={
                "manual_control_setpoint": (_STICK, [_stick(1_000_000, 0.25)]),
                "vehicle_command": (_COMMAND, [(2_000_000, 1.0, 176, 1, 255, True)]),
                "telemetry_status": (_LINK, [(3_000_000, 3, True)]),
            },
        )
    )

    with pytest.raises(RunError) as failure:
        module.fly(run, log, tmp_path)

    message = str(failure.value)
    assert (written, "rt/fmu/in/vehicle_command" in message, "2 of the replay's 3 samples" in message) == (
        ["rt/fmu/in/manual_control_input"],
        True,
        True,
    )


def test_a_replay_sends_the_samples_due_at_the_last_step_before_it_ends(tmp_path, monkeypatch):
    """A replay sends a sample due exactly at the log's end and one due between its last two ticks.

    Given a ULog whose last row, a command, is at 3.0 s, and a run that steps 0.5 s, then the command goes out.
    Given a ULog whose last row is at 3.2 s, then the run's last step reaches 3.5 s and the command goes out.
    In both, the replay steps the run on 0.5 s past the log's end, so PX4 takes the final command before it stops.
    """
    from nexus_uav._src.peers.px4_sitl import replay as module

    sent = []
    for end in (3_000_000, 3_200_000):
        written, run = _stand_in_dds(monkeypatch, matched=("vehicle_command",), tick=0.5)
        log = module.read(
            ulog_file.write(
                tmp_path / f"flight{end}.ulg",
                topics={
                    "vehicle_command": (_COMMAND, [(1_000_000, 1.0, 176, 1, 255, True), (end, 0.0, 400, 1, 255, True)])
                },
            )
        )
        module.fly(run, log, tmp_path)
        sent.append((len(written), run.time >= end / 1e6 + 0.5))

    assert sent == [(2, True), (2, True)]


def test_a_replay_whose_run_ends_before_the_logs_end_fails_unless_the_caller_capped_the_steps(tmp_path, monkeypatch):
    """A run that ends before the log's end fails the replay, unless the caller capped the run's steps.

    Given a ULog with a command at 1 s and another at 5 s, and a run that ends at 2 s, when the replay flies, then
    it raises a run error that names the end, 2.0 s, and 1 unsent sample of 2. Given the caller capped the steps,
    then it returns, and the first command alone went out.
    """
    from nexus_uav import RunError
    from nexus_uav._src.peers.px4_sitl import replay as module

    log = module.read(
        ulog_file.write(
            tmp_path / "flight.ulg",
            topics={
                "vehicle_command": (
                    _COMMAND,
                    [(1_000_000, 1.0, 176, 1, 255, True), (5_000_000, 0.0, 400, 1, 255, True)],
                )
            },
        )
    )
    _, run = _stand_in_dds(monkeypatch, matched=("vehicle_command",), tick=0.5, ends=2.0)
    with pytest.raises(RunError) as early:
        module.fly(run, log, tmp_path)
    written_capped, run = _stand_in_dds(monkeypatch, matched=("vehicle_command",), tick=0.5, ends=2.0)
    module.fly(run, log, tmp_path, capped=True)

    assert ("2.0 s" in str(early.value), "1 of 2 samples unsent" in str(early.value), len(written_capped)) == (
        True,
        True,
        1,
    )


def test_a_replays_layer_holds_the_callers_layer_by_its_absolute_path(tmp_path, monkeypatch):
    """The replay's layer names the caller's layer by its absolute path, and fails on a path with no file.

    Given a caller's layer named `./overrides.usda` from its own folder, when the replay writes its layer in
    another folder, then the sublayer is the file's absolute path. Given a path with no file, then an error names it.
    """
    pytest.importorskip("pxr")
    from nexus_uav._src.peers.px4_sitl.replay import layer
    from tests.usd import sensor_vehicle as sv

    vehicle = sv.vehicle(tmp_path, px4=True)
    (tmp_path / "overrides.usda").write_text("#usda 1.0\n")
    (tmp_path / "replay").mkdir()
    monkeypatch.chdir(tmp_path)

    written = open(layer(tmp_path / "replay", vehicle, {}, "./overrides.usda")).read()
    with pytest.raises(FileNotFoundError) as missing:
        layer(tmp_path / "replay", vehicle, {}, "./none.usda")

    assert (f"subLayers = [@{tmp_path / 'overrides.usda'}@]" in written, "none.usda" in str(missing.value)) == (
        True,
        True,
    )


def test_a_replay_starts_px4_clean_unless_the_caller_keeps_the_store(tmp_path, monkeypatch):
    """A replay starts PX4 from nothing an earlier run stored, and keeps the store when the caller asks.

    Given a ULog and a PX4 tree named, when a script replays it with no `clean`, then the run it starts is clean.
    When the script passes `clean=False`, then the run keeps the store.
    """
    pytest.importorskip("newton")
    pytest.importorskip("pxr")
    import nexus_uav._src.api.sim as api
    from nexus_uav._src.peers.px4_sitl import replay as module
    from tests.usd import sensor_vehicle as sv

    started = []

    class Run:
        def __init__(self, vehicle, **keywords):
            started.append(keywords["clean"])

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def start(self):
            pass

        def artifacts(self):
            return {}

    px4 = tmp_path / "px4"
    px4.mkdir()
    (px4 / "Makefile").write_text("px4_sitl:\n")
    monkeypatch.setattr(api, "Sim", Run)
    monkeypatch.setattr(module, "fly", lambda run, log, tree, **kw: None)
    ulog = ulog_file.write(tmp_path / "flight.ulg", info={"ver_sw": "0" * 40})
    vehicle = sv.vehicle(tmp_path, px4=True)

    replay(ulog, vehicle, scene=sv.SCENE, px4=str(px4))
    replay(ulog, vehicle, scene=sv.SCENE, px4=str(px4), clean=False)

    assert started == [True, False]


def test_a_sample_goes_to_the_topic_of_the_message_version_the_px4_tree_states(tmp_path):
    """A sample goes to the topic PX4 names for its message's version: `_v<version>` after the name, and no
    ending for version 0 or a definition that states none.

    Given a PX4 tree whose stick message states `MESSAGE_VERSION` 2, whose command message states 0 and whose link
    status message states none, then the stick topic ends `_v2` and the other two end with their names.
    """
    from nexus_uav._src.peers.px4_sitl.replay import versions

    (tmp_path / "msg" / "versioned").mkdir(parents=True)
    (tmp_path / "msg" / "versioned" / "ManualControlSetpoint.msg").write_text(
        "uint32 MESSAGE_VERSION = 2\nuint64 timestamp\n"
    )
    (tmp_path / "msg" / "versioned" / "VehicleCommand.msg").write_text("uint32 MESSAGE_VERSION = 0\nuint64 timestamp\n")
    (tmp_path / "msg" / "TelemetryStatus.msg").write_text("uint64 timestamp\n")

    stated = versions(tmp_path)

    assert [address(Sample(0, topic, {}), 0, stated[topic]).topic for topic in sorted(stated)] == [
        "rt/fmu/in/manual_control_input_v2",
        "rt/fmu/in/telemetry_status",
        "rt/fmu/in/vehicle_command",
    ]


def test_a_log_parameter_outside_px4s_identifier_syntax_stops_the_replay_before_any_run(tmp_path):
    """A log whose parameter name isn't PX4's identifier syntax stops the replay's boot parameters, and names it.

    Given a ULog with a parameter named `X$(id)`, when the replay answers its boot parameters, then a value error
    names the parameter.
    """
    ulog = ulog_file.write(tmp_path / "flight.ulg", params={"MPC_XY_VEL_MAX": 12.0, "X$(id)": 1.0})

    with pytest.raises(ValueError) as refused:
        boot_params(ulog)

    assert "X$(id)" in str(refused.value)


def test_a_replay_reads_the_vehicle_with_the_callers_layer_and_flies_the_px4_it_picked_only_for_its_run(
    tmp_path, monkeypatch
):
    """A replay adds its agent only when no layer declares one, and names its PX4 through the variable for its
    run alone.

    Given a caller's layer that declares the agent on a prim of its own, when a script replays on a vehicle that
    declares none, then the replay's layer adds no second agent. While the run builds, `NEXUS_PX4_SITL_REF` names
    the PX4 the replay picked, and after the replay it holds what it held before.
    """
    pytest.importorskip("newton")
    pytest.importorskip("pxr")
    import os

    import nexus_uav._src.api.sim as api
    from nexus_uav._src.peers.px4_sitl import replay as module
    from tests.usd import sensor_vehicle as sv

    seen = {}

    class Run:
        def __init__(self, vehicle, **keywords):
            seen["layer"] = open(keywords["layer"]).read()
            seen["variable"] = os.environ.get("NEXUS_PX4_SITL_REF")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def start(self):
            pass

        def artifacts(self):
            return {}

    px4 = tmp_path / "px4"
    px4.mkdir()
    (px4 / "Makefile").write_text("px4_sitl:\n")
    monkeypatch.setattr(api, "Sim", Run)
    monkeypatch.setattr(module, "fly", lambda run, log, tree, **kw: None)
    monkeypatch.delenv("NEXUS_PX4_SITL_REF", raising=False)
    callers = tmp_path / "callers.usda"
    callers.write_text(
        '#usda 1.0\nover "vehicle"\n{\n    def Scope "Bridge" (\n'
        '        prepend apiSchemas = ["NexusXrceAgentAPI"]\n    )\n    {\n    }\n}\n'
    )
    ulog = ulog_file.write(tmp_path / "flight.ulg", info={"ver_sw": "0" * 40})

    replay(ulog, sv.vehicle(tmp_path, px4=True), scene=sv.SCENE, px4=str(px4), layer=str(callers))

    assert ('def Scope "XrceAgent"' in seen["layer"], seen["variable"], os.environ.get("NEXUS_PX4_SITL_REF")) == (
        False,
        str(px4),
        None,
    )


def _steps(ulog) -> list[tuple[float, int, int]]:
    """Each change of PX4's arming state or mode in `ulog`: its time in seconds, then the two states."""
    from pyulog import ULog

    status = ULog(ulog, ["vehicle_status"]).get_dataset("vehicle_status").data
    steps = []
    for t, arming, nav in zip(status["timestamp"], status["arming_state"], status["nav_state"], strict=True):
        if not steps or steps[-1][1:] != (int(arming), int(nav)):
            steps.append((round(float(t) / 1e6, 3), int(arming), int(nav)))
    return steps


def _stick_positions(ulog) -> set[tuple[float, float, float, float]]:
    """The stick positions `ulog` holds in its valid rows: roll, pitch, yaw and throttle."""
    from pyulog import ULog

    sticks = ULog(ulog, ["manual_control_setpoint"]).get_dataset("manual_control_setpoint").data
    axes = zip(sticks["roll"], sticks["pitch"], sticks["yaw"], sticks["throttle"], sticks["valid"], strict=True)
    return {tuple(round(float(v), 3) for v in row[:4]) for row in axes if row[4]}


@pytest.mark.px4_sitl
@pytest.mark.xdist_group("px4_ports")
@_NEEDS_TREE
def test_a_replayed_flight_takes_each_command_arms_and_disarms_as_its_log_and_holds_the_logs_stick_rows(
    px4_vehicle, tmp_path
):
    """A replayed flight takes each command within 0.05 s of its log, arms on its sticks within 0.3 s and disarms
    after landing within 1 s, and its ULog holds the log's stick rows unchanged.

    Given the ULog of a flight the pinned PX4 flew in SITL on sticks, armed by the stick gesture in Altitude mode,
    climbed 2 m, switched to Position mode and landed, when a script replays it on the same vehicle with no PX4
    named, then the replay's ULog holds the same six arming and mode steps and the same stick positions. A step a
    replayed command drives, each of the three mode changes, is within 0.05 s of the log's. The arming, which the
    sticks drive, is within 0.3 s: the log holds the sticks at 5 Hz. The first status and the disarm after
    touchdown, which the sim's flight produces, are within 1 s.
    """
    pytest.importorskip("pymavlink")
    import nexus_uav as nx
    from nexus_uav.px4 import OffboardClient
    from tests.usd import sensor_vehicle as sv

    with nx.Sim(px4_vehicle, scene=sv.SCENE, clean=True) as sim:
        sim.start(timeout=120.0)
        with OffboardClient.connect(sim, 30.0) as op:
            op.set_mode("Altitude")
            op.arm(gesture=True)
            sim.wait_until(op.is_armed, sim_timeout=120.0)
            ground = op.relative_altitude() or 0.0
            op.set_rc(throttle=0.8)
            sim.wait_until(lambda: (op.relative_altitude() or ground) - ground >= 2.0, sim_timeout=60.0)
            op.set_rc(throttle=0.5)
            op.set_mode("Position")
            sim.sleep(2.0)
            op.land()
            sim.wait_until(lambda: not op.is_armed(), sim_timeout=120.0)
            sim.sleep(1.0)
    flown = sim.artifacts()["ulog"]

    replayed = replay(flown, px4_vehicle, scene=sv.SCENE, run_dir=tmp_path / "replay").get("ulog")

    log, again = _steps(flown), _steps(replayed) if replayed else []
    # The six steps in order: the first status, Altitude, the arming, Position, Land and the disarm.
    bounds = (1.0, 0.05, 0.3, 0.05, 0.05, 1.0)
    in_step = len(log) == len(again) == len(bounds) and all(
        a[1:] == b[1:] and abs(a[0] - b[0]) <= bound for a, b, bound in zip(log, again, bounds, strict=True)
    )
    sticks = _stick_positions(replayed) if replayed else set()
    assert (in_step, sticks) == (True, _stick_positions(flown)), f"the log's steps {log}, the replay's {again}"
