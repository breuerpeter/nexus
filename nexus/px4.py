"""PX4's public names: what a script that commands PX4 imports.

:class:`Px4Offboard` is the host end of the PX4 Software In The Loop (SITL) peer's offboard link.
The run owns the link's address and names it in its port map, so a script opens the client itself
once the sim has started. Entering the client returns at once, and the script steps the sim while
it waits for PX4's heartbeat, since PX4 runs on the sim's clock::

    with na.Sim("astro_max_base", scene="empty") as sim:
        sim.start()
        link = sim.ports["offboard"]
        with Px4Offboard(f"udpin:0.0.0.0:{link['port']}", system_id=link["system_id"]) as op:
            sim.wait_until(lambda: op.connected, sim_timeout=10.0)
            op.takeoff(2.0)
            sim.wait_until(op.at_target, sim_timeout=60.0)

:class:`Plan`, :class:`MissionItem` and :func:`read_plan` describe the mission ``upload_mission``
hands PX4, with the ``NAV_*`` and ``FRAME_*`` MAVLink constants an item names. Each lives in the
peer's folder; this module only re-exports them, so the generic top level, ``nexus``, names nothing
of PX4.
"""

from nexus._src.peers.px4_sitl.offboard import Px4Offboard
from nexus._src.peers.px4_sitl.qgc_plan import (
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
    "Plan",
    "Px4Offboard",
    "read_plan",
]
