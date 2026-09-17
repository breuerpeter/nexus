#!/usr/bin/env python3
# Acronyms: Return to Launch (RTL), Real-Time Factor (RTF).
"""Fly a QGroundControl ``.plan`` under PX4 as AUTO.MISSION.

**The shape.** The sim owns the autopilot: ``na.Sim`` builds PX4 SITL, serves the HIL link on
:4560, starts the PX4 container against it, and kills it again on the way out. What is left here is
the mission profile as a plain script against ``sim.operator``: upload the plan over MAVLink
:14540, engage AUTO.MISSION, and wait for PX4 to fly it.

**The plan is the source of truth.** ``box.plan`` beside this file is a real QGC plan, openable and
editable in QGroundControl, and it flies verbatim: its items go on the wire in the geodetic frame it
authored them in, and the sim's geodetic origin is taken *from the plan* so PX4's "first waypoint far
away from home" pre-arm check is satisfied whatever plan is loaded. Point ``PLAN`` at another file
and that mission flies instead, which is the whole point of the file format.

The mission is a 100 m box at 40 m: takeoff, four waypoints, return to launch. Both axes are
exercised on purpose, because a world-to-geodetic reflection shows up as a runaway east leg (#61).

A ZERO-arg script (the configuration lives here); only the host-specific paths ride env, ``PX4_DIR``
(default ~/code/px4) and ``PX4_IMAGE`` (the px4-sitl image), plus an optional ``--timeout`` flag.
Needs docker + a PX4 checkout + a CUDA host; the checkout does not have to be built already, because
the sim builds it before the run starts.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import nexus as na
from nexus._src.operator.qgc_plan import NAV_WAYPOINT, read_plan
from nexus.examples._lib import dump_run
from nexus.examples.controllers.px4.log_warnings import px4_warnings

PLAN = pathlib.Path(__file__).with_name("box.plan")  # the mission flown; swap the file, fly another
SCENE = "empty"  # flat ground: this example proves the mission path, not a place
DEFAULT_TIMEOUT_S = 200.0  # --timeout default: budget for PX4 to boot, wall-clock, and to reach lockstep
UPLOAD_TIMEOUT_S = 60.0  # budget [sim s] for the MISSION_COUNT, MISSION_ITEM, MISSION_ACK handshake to settle
MISSION_TIMEOUT_S = 400.0  # budget [sim s] to fly every waypoint, about 400 m of path
RTL_TIMEOUT_S = 200.0  # budget [sim s] for the plan's final return-to-launch to land


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S,
        help="budget [s] for PX4 to boot (wall-clock) and then to reach lockstep",
    )  # fmt: skip
    # parse_known_args: the launcher passes the shared diagnostics flags such as --profile through.
    args, _ = ap.parse_known_args()

    plan = read_plan(PLAN)
    waypoints = [it.seq for it in plan.items if it.command == NAV_WAYPOINT]
    stats = {
        "uploaded": False,  # PX4 accepted the mission with a MISSION_ACK
        "mission_items": len(plan.items),
        "waypoints_reached": 0,  # of len(waypoints), gated
        "landed": False,  # the plan's final RTL brought it home
        "px4_warnings": 0,  # reported, not gated: flight.py owns the warnings gate
        "rtf": 0.0,  # steady with-PX4 lockstep RTF, compile and warmup excluded
    }
    flight_ok = False
    sim = None
    op = None
    t_gcs = 0.0

    try:
        lat, lon, _ = plan.home
        # lat/lon only, deliberately: the plan's home altitude is relative to mean sea level, while
        # Sim(geo=…) reads its third field as a WGS84 ELLIPSOIDAL height. Feeding one to the other
        # offsets the origin.
        # Every altitude in the plan is relative to home, so no absolute altitude is ever needed.
        na.logger.info(f"[mission] {PLAN.name}: {len(plan.items)} items, home {lat:.6f},{lon:.6f}")
        with na.Sim(control="px4-sitl", scene=SCENE, geo=f"{lat},{lon}", device="cuda", log=True) as sim:
            sim.start(timeout=args.timeout)  # drive setup as far as PX4 lockstep
            na.logger.info("[mission] PX4 lockstep established: uploading the plan over :14540")

            # sim.operator builds lazily on first access, so this must come after start().
            t_gcs = time.time()
            op = sim.operator
            op.upload_mission(plan)
            sim.wait_until(op.mission_uploaded, sim_timeout=UPLOAD_TIMEOUT_S)
            na.logger.info(f"[mission] PX4 accepted {op.mission_count()} items: engaging AUTO.MISSION")

            # Mission mode only after the ack: PX4 refuses the mode while it has no valid mission.
            # start_mission owns the mode-before-arm ordering; PX4 climbs on the plan's NAV_TAKEOFF.
            op.start_mission()
            sim.wait_until(op.mission_complete, sim_timeout=MISSION_TIMEOUT_S)
            na.logger.info(f"[mission] flew {len(waypoints)} waypoints: waiting out the plan's RTL")

            # mission_complete reports the last WAYPOINT; the plan's final RTL item hands over to
            # RTL rather than reporting an arrival, so the landing is what closes the flight.
            sim.wait_until(lambda: op.landed_state() == "ON_GROUND", sim_timeout=RTL_TIMEOUT_S)
            flight_ok = True
    except (TimeoutError, RuntimeError) as e:
        na.logger.info(f"[mission] FAIL: {e}")
    finally:
        if op is not None:
            # Measured off PX4's own telemetry, on the failure path too, so a run that timed out
            # still reports how far it actually got.
            stats.update(
                uploaded=op.mission_uploaded(),
                waypoints_reached=sum(1 for s in waypoints if s <= op.mission_reached()),
                landed=op.landed_state() == "ON_GROUND",
            )
            if not stats["uploaded"] and op.mission_ack() is not None:
                na.logger.info(f"[mission] PX4 REJECTED the mission: MISSION_ACK type {op.mission_ack()}")
        if sim is not None:
            warn_lines = px4_warnings(sim.artifacts()["px4_log"])
            stats["px4_warnings"] = len(warn_lines)
            stats["rtf"] = round(float(sim.results().get("rtf", 0.0)), 2)
            na.logger.info(f"[mission] {time.time() - t_gcs:.0f} s of GCS time")
            dump_run(sim, "px4_mission", stats=stats)  # evaluation artifacts: flown trajectory + stats
    ok = flight_ok and stats["waypoints_reached"] == len(waypoints)
    na.logger.info(f"[mission] {json.dumps(stats)}  ->  {'MISSION OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
