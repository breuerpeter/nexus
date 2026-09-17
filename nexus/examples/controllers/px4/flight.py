#!/usr/bin/env python3
# Acronyms: Software In The Loop (SITL), Hardware In The Loop (HIL), North East Down (NED), Real Time Factor (RTF), Global Positioning System (GPS).
"""PX4 SITL flight gate: fly the Astro Max under PX4 *closed-loop* control and assert it arms, takes
off, yaws and flies a long east-west leg, with no PX4 warnings and no sim-speed regression.

**The shape.** The sim owns the autopilot: ``na.Sim`` builds PX4 SITL, serves the HIL link on :4560,
starts the PX4 container against it, and kills it again on the way out. What is left here is THE one
PX4 flight profile, written as a plain script against ``sim.operator`` (takeoff to ``TAKEOFF_ALT``,
then yaw sweeps at the hold position, then ``EAST_LEG_M`` east and back, over MAVLink :14540), plus
the sim-side gates and the evaluation dump. This script drives the sim: every operator verb returns
immediately and every wait is a ``sim.wait_until``, which steps the sim until PX4's own telemetry
says the verb landed.

The east-west leg is a regression guard (GH #61). The world to geodetic axis map was a reflection,
so long east legs diverged, and no profile here flew far enough east to notice; the yaw sweeps do
not translate at all, and the CI benchmark mission's east legs are 20 m.

The actuator model comes from the vehicle USD (authored on the rotor joints); there is no actuator
knob; whatever the vehicle authors is what flies.

A ZERO-arg script (the flight configuration lives here); only the host-specific paths ride env:
``PX4_DIR`` (default ~/code/px4) and ``PX4_IMAGE`` (the px4-sitl image), single defaults in
``nexus._src.vehicle.controllers.px4.sitl``, plus an optional ``--timeout`` flag (wall-clock
arm+climb budget). Needs docker + a PX4 checkout + a CUDA host; the checkout does not have to be
built already, because the sim builds it before the run starts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np

import nexus as na
from nexus.examples._lib import dump_run
from nexus.examples.controllers.px4.log_warnings import px4_warnings

TAKEOFF_ALT = 5.0  # takeoff altitude [m]; sets MIS_TAKEOFF_ALT, authoritative for Takeoff mode
YAW_SWEEP = (90.0, 180.0, 270.0, 0.0)  # yaw headings [deg] flown at the hold position, compass/NED
YAW_DWELL_S = 4.0  # settle per heading [sim s]; keeps the CI flight short while still exercising yaw
YAW_TIMEOUT_S = 60.0  # per-heading budget [sim s] to come round to the commanded heading
EAST_LEG_M = 200.0  # east-west out-and-back [m]; long enough that a mirrored east axis diverges, #61
LEG_TIMEOUT_S = 120.0  # per-leg budget [sim s]
LEG_ERR_MAX_M = 5.0  # gate: world-frame miss at each leg endpoint; PX4 arrives within 2 m of its own estimate
DEFAULT_TIMEOUT_S = 200.0  # --timeout default: budget for PX4 to boot, wall time, and to arm + climb, sim time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S,
        help="budget [s] for PX4 to boot (wall-clock) and then to arm and climb (sim time)",
    )  # fmt: skip
    # parse_known_args: the launcher passes the shared diagnostics flags (--profile/...) through.
    args, _ = ap.parse_known_args()
    timeout = args.timeout
    stats = {
        "takeoff_confirmed": False,
        "climb_m": 0.0,
        "time_to_takeoff_s": 0.0,
        "px4_warnings": 0,  # warning- and error-level lines from PX4 about the sim; must be 0, gated
        "east_leg_reached": False,  # both legs of the east-west out-and-back completed; gated
        "east_err_m": 0.0,  # worst world-frame miss at a leg endpoint [m]: the #61 gate
        "rtf": 0.0,  # steady with-PX4 lockstep RTF, compile/warmup-excluded, from the orchestrator
    }
    flight_ok = False
    warn_lines: list[str] = []
    sim = None
    spawn_alt: float | None = None  # set once the flight starts; the climb measurement uses it as the baseline
    t_gcs = 0.0

    try:
        # 1. The sim, in-process: it builds PX4, serves the HIL link on :4560, starts the PX4
        #    container against it, and records the run.
        na.logger.info("[sitl] Newton starting in-process (actuator from USD); serving :4560 …")
        with na.Sim(control="px4-sitl", device="cuda", log=True) as sim:
            sim.start(timeout=timeout)  # drive setup as far as PX4 lockstep
            na.logger.info("[sitl] PX4 lockstep established: flying the profile over :14540")

            # 2. The one PX4 flight profile, as a plain script: Takeoff mode, then yaw sweeps.
            #    sim.operator builds lazily on first access, so this must come after start().
            t_gcs = time.time()
            spawn_alt = sim.physics[sim.base_body].latest().altitude_m
            op = sim.operator
            op.takeoff(TAKEOFF_ALT)  # returns at once; the operator's pump arms once armable
            sim.wait_until(op.at_target, sim_timeout=timeout)
            na.logger.info(f"[sitl] climbed to {op.relative_altitude():.2f} m: yaw sweeps {list(YAW_SWEEP)}")

            # The yaw phase: hold the arrival position and altitude, sweeping the heading. The first
            # goto anchors the local frame here, so (0, 0, alt) IS the hold position, and the
            # world position right now IS that anchor, which every goto below is relative to.
            anchor_x, anchor_y, _ = sim.physics[sim.base_body].latest().position
            for heading in YAW_SWEEP:
                # YAW_SWEEP is compass, PX4/NED, clockwise from north; goto takes world yaw, which
                # runs the other way, so negate here and PX4 receives exactly the old headings.
                op.goto((0.0, 0.0, TAKEOFF_ALT), yaw=-math.radians(heading))
                sim.wait_until(op.at_target, sim_timeout=YAW_TIMEOUT_S)
                sim.sleep(YAW_DWELL_S)
                na.logger.info(f"[sitl] yaw → {heading:.0f}°")

            # The translation phase: 200 m east and back. World +y is west, so east is -y. This
            # is the #61 regression guard: under a mirrored east axis PX4's guidance and its GPS
            # disagree in sign, so the vehicle either runs away or "arrives" hundreds of metres
            # the other way. Measured against ground truth, because a mirrored map lets PX4
            # believe it got there.
            for leg in ((0.0, -EAST_LEG_M, TAKEOFF_ALT), (0.0, 0.0, TAKEOFF_ALT)):
                op.goto(leg)
                try:
                    sim.wait_until(op.at_target, sim_timeout=LEG_TIMEOUT_S)
                finally:
                    # In the finally, so a leg that times out still records how far it got. Under
                    # a mirrored map that miss IS the diagnostic, and reporting 0.0 there would
                    # read as a clean flight on the one metric whose job is to catch it.
                    x, y, _ = sim.physics[sim.base_body].latest().position
                    err = math.hypot(x - (anchor_x + leg[0]), y - (anchor_y + leg[1]))
                    stats["east_err_m"] = round(max(stats["east_err_m"], err), 2)
                    na.logger.info(f"[sitl] leg {leg}: world miss {err:.2f} m")
            stats["east_leg_reached"] = True
            flight_ok = True
    except (TimeoutError, RuntimeError) as e:
        na.logger.info(f"[sitl] takeoff FAIL: {e}")
    finally:
        if sim is not None:
            # Warnings gate: a clean SITL flight must produce no PX4 warnings about the sim, for example no
            # "Strong magnetic interference" warning. The PX4 console is an artifact of the run now, so the
            # gate reads the log of the container this sim started.
            warn_lines = px4_warnings(sim.artifacts()["px4_log"])
            stats["px4_warnings"] = len(warn_lines)
            if warn_lines:
                shown = sorted(set(warn_lines))
                na.logger.info(f"[sitl] {len(warn_lines)} PX4 warning line(s), FAIL ({len(shown)} distinct):")
                for w in shown[:20]:
                    na.logger.info(f"  {w}")
            # Sim-side truth: the climb from the recorder history, the max altitude over the whole
            # flight. Measured here rather than on the happy path so a flight that timed out still
            # reports how far it actually got.
            if spawn_alt is not None:
                alts = np.array([s.altitude_m for s in sim.physics[sim.base_body].history()])
                climb = float(alts.max() - spawn_alt) if alts.size else 0.0
                stats.update(
                    takeoff_confirmed=flight_ok,
                    climb_m=round(climb, 2),
                    time_to_takeoff_s=round(time.time() - t_gcs, 1),
                )
                na.logger.info(f"[sitl] flight profile {'OK' if flight_ok else 'FAIL'}, climbed {climb:.2f} m")
            stats["rtf"] = round(float(sim.results().get("rtf", 0.0)), 2)
            dump_run(sim, "px4_sitl", stats=stats)  # evaluation artifacts: flown trajectory + stats
    # east_err_m is the load-bearing gate, not east_leg_reached: a mirrored east axis lets PX4
    # believe it arrived while the airframe sits hundreds of metres the other way, GH #61.
    ok = flight_ok and not warn_lines and stats["east_err_m"] <= LEG_ERR_MAX_M
    na.logger.info(f"[sitl] {json.dumps(stats)}  ->  {'FLIGHT OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
