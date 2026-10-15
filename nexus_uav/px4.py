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
peer's folder; this module only re-exports them, so the generic top level, ``nexus_uav``, names nothing
of PX4.
"""

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
]
