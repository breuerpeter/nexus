"""The replay of a real flight's ULog into PX4 Software In The Loop (SITL), #429.

A script calls ``nexus_uav.px4.replay`` with a flight's ULog, a vehicle and a scene. The run flies PX4 SITL on the
inputs the log holds, each sent as its topic over PX4's bridge to the Data Distribution Service (DDS) at its
logged time, and its artifacts hold PX4's own ULog. The ULog is the only source of what the replay sends.

The replay changes nothing in PX4. It flies the commit the log names, unless the caller names another PX4.
PX4 boots on the log's parameters, and the replay's override layer authors them on the vehicle's PX4 peer and
declares the Micro XRCE-DDS agent peer, the bridge's other end. The replay then steps the run to the log's
end, and sends each sample once the run reaches its time.

The replay sends what the bridge lists as an input: the sticks, the external commands and the link status. It
leaves out what the bridge lacks: the switches, the gimbal command and the air traffic.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import itertools
import os
import re
import statistics
import subprocess
import time
import types
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

from nexus_uav._src.core import logger
from nexus_uav._src.core.errors import RunError
from nexus_uav._src.peers import pins
from nexus_uav._src.peers.pins import PeerPin

from . import OFFBOARD_PORT, checkout

SKIPPED = {
    "CAL_*": "the real sensors' calibration: PX4 reports the simulated accelerometer, gyro and compass uncalibrated",
    "SENS_*": "the real board's sensor setup: PX4 reports an attitude failure",
    "SYS_HAS_*": "the hardware the real vehicle carries: PX4 reports a missing distance sensor",
    "COM_POWER_COUNT": "a count above 1 denies arming: SITL has one power source",
    "COM_CPU_MAX": "the real board's load limit: PX4 reports the CPU load too high in a run faster than real time",
    "CBRK_SUPPLY_CHK": "the real vehicle's supply check: PX4 reports the system power unavailable",
    "UXRCE_DDS_*": "the real vehicle's bridge setup: no replayed sample reaches PX4",
    "MAV_SYS_ID": "the real vehicle's system id: PX4 on an instance other than 0 ignores each replayed command",
}
"""Each parameter of a log that PX4 SITL keeps as its own, a name or a pattern of names, with the measured reason."""

# The bridge input each replayed topic of a log goes to, and PX4's message for that input.
_INPUTS = {
    "manual_control_setpoint": ("manual_control_input", "ManualControlSetpoint"),
    "vehicle_command": ("vehicle_command", "VehicleCommand"),
    "telemetry_status": ("telemetry_status", "TelemetryStatus"),
}
_MESSAGES = dict(_INPUTS.values())
_PEER_SCHEMA = "NexusPx4SitlAPI"
_AGENT_SCHEMA = "NexusXrceAgentAPI"
_FROM_LOG = "ver_sw"  # the end of the source of a pin the log names
_MATCH_HOLD_S = 2.0  # wall seconds the run holds for a due sample whose writer no reader has matched
_EXTRA = "the replay needs the extra nexus-uav[replay], which installs pyulog and cyclonedds"


class Sample(NamedTuple):
    """One row the replay sends to PX4.

    Args:
        time: The row's logged time in microseconds, on PX4's clock.
        topic: The bridge input the row goes to, such as ``manual_control_input``.
        values: The row's fields by name, as the log holds them.
    """

    time: int
    topic: str
    values: Mapping[str, object]


def read(ulog: str | os.PathLike):
    """The log ``ulog`` names, read whole.

    Raises:
        ImportError: The extra that reads a ULog isn't installed; the message names it.
    """
    try:
        from pyulog import ULog
    except ImportError as exc:
        raise ImportError(_EXTRA) from exc
    return ULog(str(ulog))


def _rows(log, topic: str) -> list[dict]:
    """Each row of ``topic`` in ``log``, of every instance, in time order: its fields by name."""
    rows = []
    for data in log.data_list:
        if data.name != topic:
            continue
        columns = {name: column.tolist() for name, column in data.data.items()}
        rows += [dict(zip(columns, values, strict=True)) for values in zip(*columns.values(), strict=True)]
    return sorted(rows, key=lambda row: row["timestamp"])


def _held(rows: list[dict]) -> list[dict]:
    """The valid stick rows of ``rows``, with copies of a row across the gap after it while the next row is valid.

    A log holds a subsample of the sticks and drops rows, so a gap in the rows is no stick loss. A copy takes its own
    time, one period of the rows after the last, and the period is the rows' median.
    """
    if len(rows) > 1:
        period = int(statistics.median(b["timestamp"] - a["timestamp"] for a, b in itertools.pairwise(rows)))
    else:
        period = 0
    sent = []
    for row, after in itertools.zip_longest(rows, rows[1:]):
        if not row["valid"]:
            continue
        sent.append(row)
        if period <= 0 or after is None or not after["valid"]:
            continue
        t = row["timestamp"] + period
        while t < after["timestamp"] - period // 2:
            stamps = {key: t for key in ("timestamp", "timestamp_sample") if key in row}
            sent.append({**row, **stamps})
            t += period
    return sent


def log_samples(log) -> list[Sample]:
    """What :func:`samples` answers, from a log read already."""
    sent = [
        Sample(row["timestamp"], "manual_control_input", row) for row in _held(_rows(log, "manual_control_setpoint"))
    ]
    sent += [
        Sample(row["timestamp"], "vehicle_command", row)
        for row in _rows(log, "vehicle_command")
        if row.get("from_external")
    ]
    sent += [Sample(row["timestamp"], "telemetry_status", row) for row in _rows(log, "telemetry_status")]
    return sorted(sent, key=lambda sample: sample.time)


def samples(ulog: str | os.PathLike) -> list[Sample]:
    """The rows a replay of ``ulog`` sends, in time order: each valid stick row, each external command and
    each link status row. Across a gap in the stick rows, copies of the last row fill the gap while the next
    row is valid. A topic the log lacks sends nothing.
    """
    return log_samples(read(ulog))


def address(sample: Sample, instance: int) -> Sample:
    """``sample`` as PX4 instance ``instance`` takes it: on that instance's topic, and a command with that
    instance's system id as its target.

    PX4 prefixes the topics of an instance other than 0 with ``px4_<instance>``, and takes ``instance + 1`` as
    its system id.
    """
    prefix = f"px4_{instance}/" if instance else ""
    values = sample.values
    if sample.topic == "vehicle_command":
        values = {**values, "target_system": instance + 1}
    return Sample(sample.time, f"rt/{prefix}fmu/in/{sample.topic}", values)


def log_pin(log, ulog, px4: str | None, default: PeerPin) -> PeerPin:
    """What :func:`px4_pin` answers, from a log read already."""
    if px4 is not None:
        return PeerPin(ref=px4, source="the replay's px4")
    if default.source == pins.variable(checkout.NAME):
        return default  # the variable names a PX4 for every run of this environment
    commit = log.msg_info_dict.get("ver_sw")
    if not commit:
        raise ValueError(f"{ulog} names no PX4 commit, ver_sw; name the PX4 to fly with px4")
    named = checkout.parse(default)
    if isinstance(named, Path):
        raise ValueError(
            f"{default.source} pins a checkout, which names no repository for the log's commit {commit}; "
            "name the PX4 to fly with px4"
        )
    return PeerPin(ref=f"{named.repo}@{commit}", source=f"{ulog}: {_FROM_LOG}")


def px4_pin(ulog: str | os.PathLike, px4: str | None, default: PeerPin) -> PeerPin:
    """The PX4 a replay of ``ulog`` flies: ``px4`` when the caller names one, else the commit the log names,
    from the repository of ``default``, the vehicle's PX4 pin. A pin the peer's variable set stays.

    Raises:
        ValueError: The log names no commit, or ``default`` names a checkout and so no repository.
    """
    return log_pin(read(ulog), ulog, px4, default)


def log_params(log, delta: Mapping[str, float] | None) -> dict[str, float]:
    """What :func:`boot_params` answers, from a log read already."""
    params = {
        name: value
        for name, value in log.initial_parameters.items()
        if not any(fnmatch.fnmatchcase(name, pattern) for pattern in SKIPPED)
    }
    params.update(delta or {})
    # With the sync on, PX4 shifts each incoming timestamp by an offset that drifts once the run is faster than
    # real time. It reads the parameter at boot.
    params["UXRCE_DDS_SYNCT"] = 0
    return params


def boot_params(ulog: str | os.PathLike, delta: Mapping[str, float] | None = None) -> dict[str, float]:
    """The parameters PX4 boots a replay of ``ulog`` on: the log's list without the skipped ones, :data:`SKIPPED`,
    then ``delta``, then ``UXRCE_DDS_SYNCT`` 0.
    """
    return log_params(read(ulog), delta)


def tree(pin: PeerPin) -> Path:
    """The PX4 tree ``pin`` names, fetched when this machine lacks it.

    Raises:
        ValueError: The fetch of a log's own commit failed; the message names the commit and the keyword that
            names another PX4.
        subprocess.CalledProcessError: The fetch of a PX4 the caller named failed.
    """
    try:
        return checkout.tree(pin)
    except subprocess.CalledProcessError as exc:
        if not pin.source.endswith(_FROM_LOG):
            raise
        repo, commit = pin.ref.split("@", 1)
        raise ValueError(
            f"the log's PX4 commit {commit} couldn't be fetched from {repo}; name the PX4 to fly with px4, "
            "a repo@sha or the path of a checkout"
        ) from exc


def layer(folder: Path, vehicle_file: str, params: Mapping[str, float], under: str | None) -> str:
    """Write the replay's override layer in ``folder`` and return its path: the boot parameters on the
    vehicle's PX4 peer, and the agent peer on a scope of its own when the vehicle declares none.

    Raises:
        ValueError: The vehicle declares no PX4 peer.
    """
    from nexus_uav._src.usd.reader import read_declarations

    declared = {schema: prim for prim, schema, _ in read_declarations(vehicle_file)}
    if _PEER_SCHEMA not in declared:
        raise ValueError(f"{vehicle_file} declares no PX4 SITL peer, {_PEER_SCHEMA}, for the replay to fly")
    root, _, peer = declared[_PEER_SCHEMA].strip("/").partition("/")
    entries = ", ".join(f'"{name}={value}"' for name, value in params.items())
    agent = ""
    if _AGENT_SCHEMA not in declared:
        agent = (
            f'    def Scope "XrceAgent" (\n        prepend apiSchemas = ["{_AGENT_SCHEMA}"]\n    )\n    {{\n    }}\n'
        )
    sublayers = f"(\n    subLayers = [@{under}@]\n)\n" if under else ""
    opened = "".join(
        f'{"    " * (depth + 1)}over "{name}"\n{"    " * (depth + 1)}{{\n' for depth, name in enumerate(peer.split("/"))
    )
    closed = "".join(f"{'    ' * (depth + 1)}}}\n" for depth in reversed(range(len(peer.split("/")))))
    attribute = f"{'    ' * (len(peer.split('/')) + 1)}string[] nexus:params = [{entries}]\n"
    path = folder / "replay.usda"
    path.write_text(f'#usda 1.0\n{sublayers}\nover "{root}"\n{{\n{opened}{attribute}{closed}{agent}}}\n')
    return str(path)


def _message_type(px4_tree: Path, name: str):
    """PX4's message ``name`` as a DDS struct, from the tree's definition, its fields in the file's order."""
    from cyclonedds.idl import IdlStruct
    from cyclonedds.idl import types as T

    primitive = {
        "bool": bool,
        "char": T.char,
        "float32": T.float32,
        "float64": T.float64,
        **{f"{sign}int{bits}": getattr(T, f"{sign}int{bits}") for sign in ("", "u") for bits in (8, 16, 32, 64)},
    }
    source = px4_tree / "msg" / f"{name}.msg"
    if not source.exists():
        source = px4_tree / "msg" / "versioned" / f"{name}.msg"
    fields = []
    for line in source.read_text().splitlines():
        statement = line.split("#")[0].strip()
        if not statement or "=" in statement:  # a comment or a constant
            continue
        kind, field = statement.split()[:2]
        array = re.fullmatch(r"(\w+)\[(\d+)\]", kind)
        fields.append((field, T.array[primitive[array[1]], int(array[2])] if array else primitive[kind]))
    cls = types.new_class(
        name,
        (IdlStruct,),
        {"typename": f"px4_msgs::msg::dds_::{name}_"},
        lambda namespace: namespace.update(__annotations__=dict(fields)),
    )
    return dataclasses.dataclass(cls)


def _message(cls, values: Mapping[str, object]):
    """A message of type ``cls`` from a logged row: a field the log lacks is zero."""
    built = {}
    for field in dataclasses.fields(cls):
        if field.name in values:
            value = values[field.name]
            built[field.name] = bool(value) if field.type is bool else value
        else:  # an array, whose items a log names field[i], or a field the log's firmware lacks
            items = [values[key] for key in sorted(values, key=_index) if key.startswith(f"{field.name}[")]
            built[field.name] = items or (False if field.type is bool else 0)
    return cls(**built)


def _index(key: str) -> int:
    """The index of an array item's name, ``field[3]``, and 0 for any other name."""
    found = re.search(r"\[(\d+)\]$", key)
    return int(found[1]) if found else 0


def fly(run, log, px4_tree: Path) -> None:
    """Step the started run ``run`` to the end of ``log``, and send each sample once the run reaches its time.

    Args:
        run: The started ``Sim`` whose PX4 the replay feeds.
        log: The log, from :func:`read`.
        px4_tree: The PX4 tree the run flies, which defines the messages.

    Raises:
        ImportError: The extra that writes to DDS isn't installed; the message names it.
        RunError: A due sample never reached PX4, since no reader matched its writer; the message names the topic
            and the count of unsent samples.
    """
    try:
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.pub import DataWriter
        from cyclonedds.topic import Topic
    except ImportError as exc:
        raise ImportError(_EXTRA) from exc
    sent = log_samples(log)
    end = log.last_timestamp / 1e6
    instance = run.ports["offboard"]["port"] - OFFBOARD_PORT
    participant = DomainParticipant(0)
    kinds = {topic: _message_type(px4_tree, message) for topic, message in _MESSAGES.items()}
    writers = {
        topic: DataWriter(participant, Topic(participant, address(Sample(0, topic, {}), instance).topic, kind))
        for topic, kind in kinds.items()
    }
    logger.info(f"replaying {len(sent)} samples to {end:.1f} s on PX4 instance {instance}")
    due = 0
    waited = -1  # the sample this loop has held the run for, so one sample holds it once
    blocked = False  # whether the last pass left a due sample unsent, its writer matched by no reader
    stepping = True
    while stepping and run.time < end:
        blocked = False
        while due < len(sent) and sent[due].time / 1e6 <= run.time:
            writer = writers[sent[due].topic]
            # DDS drops a sample a writer sends before a reader matches it. The agent makes its readers once
            # PX4's client starts, and the match then takes wall time, not PX4's. So the run holds for a due
            # sample once, and a sample with no reader yet goes out late, at the match.
            if not writer.get_matched_subscriptions():
                if waited < due:
                    waited = due
                    deadline = time.monotonic() + _MATCH_HOLD_S
                    while not writer.get_matched_subscriptions() and time.monotonic() < deadline:
                        time.sleep(0.01)
                if not writer.get_matched_subscriptions():
                    blocked = True
                    break
            sample = address(sent[due], instance)
            writer.write(_message(kinds[sent[due].topic], sample.values))
            due += 1
        stepping = run.step()  # false once the run has ended, at its ``max_steps``
    if blocked:
        unsent = sum(1 for sample in sent[due:] if sample.time / 1e6 <= run.time)
        raise RunError(
            f"{unsent} of the replay's {len(sent)} samples never reached PX4: no reader matched "
            f"{address(sent[due], instance).topic}; is the Micro XRCE-DDS agent peer running, and does this PX4's "
            "bridge list the topic?"
        )
