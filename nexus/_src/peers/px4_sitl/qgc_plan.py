"""Read a QGroundControl ``.plan`` file into the mission items PX4 expects on the wire.

A ``.plan`` is JSON, and its mission items are already in MAVLink's own vocabulary: each carries a
``command``, a ``frame`` and seven ``params``, where params 5, 6 and 7 are latitude, longitude and
altitude. So a plan needs translating, not interpreting: :class:`MissionItem` is a
``MISSION_ITEM_INT`` with its fields named, and the upload in ``Px4Offboard`` puts them on the wire
verbatim. Nothing here converts through the sim's world frame: the plan is geodetic and stays
geodetic, which is the whole point of flying a real file rather than scripted gotos.

This module deliberately imports no pymavlink. It's pure JSON handling, so it tests without a
MAVLink stack and the enum values it names below are the wire constants, not imported symbols.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

# The wire constants this module names in errors and docstrings, from the MAVLink common dialect.
NAV_WAYPOINT = 16
NAV_RETURN_TO_LAUNCH = 20
NAV_TAKEOFF = 22
FRAME_GLOBAL_RELATIVE_ALT = 3


@dataclass(frozen=True)
class MissionItem:
    """One mission item, in the shape ``MISSION_ITEM_INT`` carries it.

    Attributes:
        seq: Position in the mission, assigned from the item's index in the file.
        frame: MAV_FRAME for this item; ``3``, ``FRAME_GLOBAL_RELATIVE_ALT``, means ``z`` is
            the altitude in metres relative to the home position.
        command: MAV_CMD for this item, for example ``16`` for ``NAV_WAYPOINT``, ``22`` for
            ``NAV_TAKEOFF``, ``20`` for ``NAV_RETURN_TO_LAUNCH``.
        autocontinue: Whether PX4 proceeds to the next item on arrival.
        params: ``param1`` through ``param4``. A ``null`` in the file arrives as ``NaN``, which is
            how MAVLink spells "no value": ``param4`` NaN leaves the heading to PX4.
        x: Latitude in degrees times 1e7: the integer field, which is why the ``MISSION_ITEM_INT``
            variant exists.
        y: Longitude in degrees times 1e7.
        z: Altitude in metres, in this item's ``frame``.
    """

    seq: int
    frame: int
    command: int
    autocontinue: bool
    params: tuple[float, float, float, float]
    x: int
    y: int
    z: float


@dataclass(frozen=True)
class Plan:
    """A parsed ``.plan``: where the vehicle starts, and the items it flies.

    Attributes:
        home: ``plannedHomePosition`` as ``(lat [deg], lon [deg], alt [m])``. The altitude is height
            over mean sea level, which is a different datum from the WGS84 ellipsoidal height
            ``Sim(geo=…)`` takes: pass latitude and longitude across, never this altitude.
        items: The mission items in file order, with ``seq`` matching the index.
    """

    home: tuple[float, float, float]
    items: tuple[MissionItem, ...]


def _param(value) -> float:
    """A ``param1``–``param4`` field: JSON ``null`` becomes NaN, which MAVLink reads as no value."""
    return math.nan if value is None else float(value)


def _dege7(value) -> int:
    """A latitude/longitude field as the integer degE7 ``MISSION_ITEM_INT`` carries.

    ``null`` becomes ``0``, which is what a positionless item such as ``NAV_RETURN_TO_LAUNCH``
    stores in the file.
    """
    return 0 if value is None else round(float(value) * 1e7)


def read_plan(path: str | os.PathLike) -> Plan:
    """Read a QGroundControl ``.plan`` file.

    The items keep their file order and get a ``seq`` from their index, so the mission uploads as
    ``0..N-1`` with the first real item at ``seq`` 0. There is no implicit home item: PX4 takes the
    mission as sent, and this is what MAVSDK puts on the wire too.

    This reads the mission and nothing else: it skips ``geoFence`` and ``rallyPoints``.

    Args:
        path: Path to the ``.plan`` file.

    Returns:
        The parsed :class:`Plan`.

    Raises:
        ValueError: The file isn't a plan, meaning ``fileType`` isn't ``"Plan"``; it carries no
            ``mission`` object; an item isn't a ``"SimpleItem"``, since this reader doesn't handle a
            QGroundControl survey ``ComplexItem``; or an item doesn't carry exactly seven ``params``.
    """
    with open(path) as fh:
        doc = json.load(fh)

    if doc.get("fileType") != "Plan":
        raise ValueError(f"{path}: not a QGC .plan (fileType={doc.get('fileType')!r})")
    mission = doc.get("mission")
    if not isinstance(mission, dict):
        raise ValueError(f"{path}: no 'mission' object")

    items = []
    for seq, raw in enumerate(mission.get("items", [])):
        kind = raw.get("type")
        if kind != "SimpleItem":
            raise ValueError(f"{path}: item {seq} is a {kind!r}; only 'SimpleItem' is supported")
        params = raw.get("params", [])
        if len(params) != 7:
            raise ValueError(f"{path}: item {seq} has {len(params)} params, expected 7")
        items.append(
            MissionItem(
                seq=seq,
                frame=int(raw["frame"]),
                command=int(raw["command"]),
                autocontinue=bool(raw.get("autoContinue", True)),
                params=(_param(params[0]), _param(params[1]), _param(params[2]), _param(params[3])),
                x=_dege7(params[4]),
                y=_dege7(params[5]),
                z=0.0 if params[6] is None else float(params[6]),
            )
        )

    lat, lon, alt = mission.get("plannedHomePosition", [0.0, 0.0, 0.0])
    return Plan(home=(float(lat), float(lon), float(alt)), items=tuple(items))
