"""PX4's public names: what a script that commands PX4 imports.

:class:`OffboardClient` is the host end of the PX4 Software In The Loop (SITL) peer's offboard link.
The run owns the link's address and names it in its port map, so a script connects the client
itself once the sim has started. ``OffboardClient.connect`` steps the sim while it waits for PX4's
heartbeat, since PX4 runs on the sim's clock, and raises :class:`nexus_uav.RunError` when none
comes. So does any wait once the run loses PX4, which separates a broken run from a failed check::

    with nx.Sim("astro_max_base", scene="empty") as sim:
        sim.start()
        with OffboardClient.connect(sim, 10.0) as op:
            op.takeoff(2.0)
            sim.wait_until(op.at_target, sim_timeout=60.0)

:class:`Plan`, :class:`MissionItem` and :func:`read_plan` describe the mission ``upload_mission``
hands PX4, with the ``NAV_*`` and ``FRAME_*`` MAVLink constants an item names. Each lives in the
peer's folder; this module re-exports them, so the generic top level, ``nexus_uav``, names nothing
of PX4.

:func:`replay` flies PX4 on the inputs a real flight's ULog holds. It builds its own run, so it sits here,
over ``Sim``, and its parts live in the peer's folder.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nexus_uav._src.peers import pins as _pins
from nexus_uav._src.peers.px4_sitl import checkout as _checkout
from nexus_uav._src.peers.px4_sitl import replay as _replay
from nexus_uav._src.peers.px4_sitl.offboard import OffboardClient
from nexus_uav._src.peers.px4_sitl.qgc_plan import (
    FRAME_GLOBAL_RELATIVE_ALT,
    NAV_RETURN_TO_LAUNCH,
    NAV_TAKEOFF,
    NAV_WAYPOINT,
    MissionItem,
    Plan,
    read_plan,
)

__all__ = [
    "FRAME_GLOBAL_RELATIVE_ALT",
    "NAV_RETURN_TO_LAUNCH",
    "NAV_TAKEOFF",
    "NAV_WAYPOINT",
    "MissionItem",
    "OffboardClient",
    "Plan",
    "read_plan",
    "replay",
]


def replay(
    ulog: str | os.PathLike,
    vehicle: str,
    *,
    scene: str,
    params: Mapping[str, float] | None = None,
    px4: str | None = None,
    **sim: Any,
) -> dict[str, str]:
    """Replay the flight ``ulog`` holds into PX4 SITL on ``vehicle`` in ``scene``, to the log's end.

    The run flies the PX4 commit the log names, boots it on the log's parameters, and sends each stick row,
    external command and link status row of the log at its logged time, over PX4's bridge to the Data
    Distribution Service (DDS). The replay changes nothing in PX4, and the log is the only source of what it
    sends. It leaves out what the bridge lacks: the switches, the gimbal command and the air traffic::

        from nexus_uav.px4 import replay

        artifacts = replay("flight.ulg", "astro_max_base", scene="empty", params={"MPC_XY_VEL_MAX": 5.0})
        print(artifacts["ulog"])  # PX4's own log of the replayed flight

    Args:
        ulog: The flight's ULog.
        vehicle: The vehicle the run flies, a catalog name or a vehicle file's path. It declares the PX4 SITL peer.
        scene: The scene the run flies in.
        params: PX4 parameters that differ from the log's, by name.
        px4: The PX4 to fly, ``repo@sha`` or a checkout's path; the log's own commit when ``None``.
        **sim: More keywords of :class:`nexus_uav.Sim`, such as ``run_dir`` or ``layer``.

    Returns:
        The run's artifacts, PX4's own ULog among them as ``ulog``.

    Raises:
        ImportError: The extra ``nexus-uav[replay]`` isn't installed.
        ValueError: The fetch of the log's commit failed, or the log names none, or the vehicle declares no
            PX4 SITL peer.
        nexus_uav.RunError: The run failed.
    """
    from nexus_uav._src.api.sim import Sim
    from nexus_uav._src.config import LaunchConfig, resolve
    from nexus_uav._src.config.catalog import catalog_sources

    log = _replay.read(ulog)
    launch = LaunchConfig()
    launch.set_vehicle(vehicle)
    launch.set_scene(scene)
    launch.catalog = sim.get("catalog")
    default = _checkout.check(_pins.resolve(_checkout.NAME, catalog_sources(launch.catalog)))
    pin = _replay.log_pin(log, ulog, px4, default)
    tree = _replay.tree(pin)  # the fetch of the log's commit fails here, before any peer starts
    with tempfile.TemporaryDirectory(prefix="nexus-replay-") as folder:
        vehicle_file = resolve(launch, cache_dir=sim.get("cache_dir")).vehicle_usd_path
        layer = _replay.layer(Path(folder), vehicle_file, _replay.log_params(log, params), sim.pop("layer", None))
        with Sim(vehicle, scene=scene, layer=layer, pins={_checkout.NAME: pin.ref}, **sim) as run:
            run.start()
            _replay.fly(run, log, tree)
        return run.artifacts()
