"""The replay of a real flight's ULog into PX4 Software In The Loop (SITL), #429.

A script calls :func:`replay` with a flight's ULog, a vehicle and a scene. The run flies PX4 SITL on the
inputs the log holds, each sent as its topic over PX4's bridge to the Data Distribution Service (DDS) at its
logged time, and its artifacts hold PX4's own ULog. The ULog is the only source of what the replay sends.

This module holds the signatures alone. The change for #429 fills them in.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import NamedTuple

from nexus_uav._src.peers.pins import PeerPin


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


def samples(ulog: str | os.PathLike) -> list[Sample]:
    """The rows a replay of ``ulog`` sends, in time order: each valid stick row, each external command and
    each link status row. Across a gap in the stick rows, copies of the last row fill the gap while the next
    row is valid.
    """
    return []


def address(sample: Sample, instance: int) -> Sample:
    """``sample`` as PX4 instance ``instance`` takes it: on that instance's topic, and a command with that
    instance's system id as its target.
    """
    return sample


def px4_pin(ulog: str | os.PathLike, px4: str | None, default: PeerPin) -> PeerPin:
    """The PX4 a replay of ``ulog`` flies: ``px4`` when the caller names one, else the commit the log names,
    from the repository of ``default``, the vehicle's PX4 pin.
    """
    return default


def boot_params(ulog: str | os.PathLike, delta: Mapping[str, float] | None = None) -> dict[str, float]:
    """The parameters PX4 boots a replay of ``ulog`` on: the log's list without the skipped ones, then
    ``delta``, then ``UXRCE_DDS_SYNCT`` 0.
    """
    return {}


def replay(
    ulog: str | os.PathLike,
    vehicle: str,
    *,
    scene: str,
    params: Mapping[str, float] | None = None,
    px4: str | None = None,
    **sim,
) -> dict[str, str]:
    """Replay the flight ``ulog`` holds into PX4 SITL on ``vehicle`` in ``scene``, to the log's end.

    Args:
        ulog: The flight's ULog.
        vehicle: The vehicle the run flies, a catalog name or a vehicle file's path.
        scene: The scene the run flies in.
        params: PX4 parameters that differ from the log's, by name.
        px4: The PX4 to fly, ``repo@sha`` or a checkout's path; the log's own commit when ``None``.
        **sim: More keywords of :class:`nexus_uav.Sim`, such as ``run_dir``.

    Returns:
        The run's artifacts, PX4's own ULog among them as ``ulog``.
    """
    return {}
