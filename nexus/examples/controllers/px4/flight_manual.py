#!/usr/bin/env python3
# Acronyms: Software in the Loop (SITL).
"""PX4 SITL flight via the INTERNAL-STICKS path: fly the vehicle under PX4 closed-loop control from
a `MANUAL_CONTROL` (#69) stick stream, arm with the throttle-down + yaw-right gesture, throttle up,
and assert the climb. The manual counterpart to :mod:`flight` (which flies the autonomous profile).

**Same shape as flight.py.** The sim owns the autopilot: ``na.Sim`` builds PX4 SITL, serves the HIL
link on :4560, starts the PX4 container against it, and kills it again on the way out. What is left
here is a plain script against ``sim.operator``, whose sticks are streamed by the operator's own pump
over MAVLink :14540, the *same* #69 bytes the Ground Control Station (GCS) emits from decoded Pilot Pro sticks
(freeflycontroller ``PILOT_PRO_OUTPUTS_MAVLINK_MANUAL_CONTROL`` msg 52537 -> x/y/z/r). So this
verifies the gap-2 manual path end-to-end in SITL without a physical Pilot Pro or an emulated companion.
The decoder half is unit-tested; this exercises the *flight* half (does a #69 gesture arm + fly
this PX4). This script drives the sim: every operator verb returns immediately and every wait is a
``sim.wait_until``, which steps the sim until PX4's own telemetry says the verb landed.

Only the operator steps differ from flight.py: it flies in Altitude mode (ALTCTL), where
throttle-centre holds altitude and throttle-up climbs, and it arms with the stick gesture rather
than an arm command.

A ZERO-arg script (the flight configuration lives here); only the host-specific paths ride env,
``PX4_DIR`` / ``PX4_IMAGE`` as in flight.py, plus the optional ``--timeout``, ``--throttle`` and
``--climb-target`` flags. Needs docker + a PX4 checkout + a CUDA host; as in flight.py the checkout
does not have to be built already, because the sim builds it before the run starts.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

import nexus as na
from nexus.examples._lib import dump_run
from nexus.examples.controllers.px4.log_warnings import px4_warnings

DEFAULT_TIMEOUT_S = 200.0  # --timeout default: wall-clock budget for arm + climb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S,
        help="budget [s] for PX4 to boot (wall-clock) and then to arm and climb (sim time)",
    )  # fmt: skip
    ap.add_argument("--throttle", type=float, default=0.8, help="throttle stick for the climb (0..1; >0.5 climbs)")
    ap.add_argument("--climb-target", type=float, default=2.0, help="relative climb (m) that counts as success")
    # parse_known_args: the launcher passes the shared diagnostics flags such as --profile through.
    args, _ = ap.parse_known_args()

    stats = {"takeoff_confirmed": False, "climb_m": 0.0, "px4_warnings": 0, "rtf": 0.0}
    flight_ok = False
    sim = None
    spawn_alt: float | None = None  # set once the flight starts; the climb counts from it

    try:
        na.logger.info("[manual] Newton starting in-process (actuator from USD); serving :4560 …")
        with na.Sim(control="px4-sitl", device="cuda", log=True) as sim:
            sim.start(timeout=args.timeout)  # drive setup as far as PX4 lockstep
            na.logger.info("[manual] PX4 lockstep established: flying via MANUAL_CONTROL (:14540)")

            # sim.operator builds lazily on first access, so this must come after start().
            spawn_alt = sim.physics[sim.base_body].latest().altitude_m
            op = sim.operator
            # SITL's rcS already sets this, but it's a runtime default rather than a compiled-in
            # one: an upstream bump could flip it, and the only symptom would be a gesture that
            # never arms. Say it out loud. MAN_ARM_GESTURE has a compiled-in value of 1 and stays
            # alone.
            op.param_set_int("COM_RC_IN_MODE", 1)  # 1 = MAVLink only
            op.set_mode("Altitude")  # a manual mode: the operator auto-starts the neutral stick stream
            op.arm(gesture=True)  # the pump holds neutral until armable, then gestures
            na.logger.info("[manual] Altitude requested, arm gesture lodged: waiting for the motors")
            sim.wait_until(op.is_armed, sim_timeout=args.timeout)

            alt0 = op.relative_altitude() or 0.0
            na.logger.info(f"[manual] ARMED (by stick gesture): throttle to {args.throttle:.2f}")
            op.set_rc(throttle=args.throttle)  # throttle up in ALTCTL
            sim.wait_until(
                lambda: (op.relative_altitude() or alt0) - alt0 >= args.climb_target, sim_timeout=args.timeout
            )
            op.set_rc(throttle=0.5)  # ease back to altitude hold
            flight_ok = True
            na.logger.info(f"[manual] TAKEOFF via MANUAL_CONTROL: PX4 reports {op.relative_altitude():.2f} m")
    except (TimeoutError, RuntimeError) as e:
        na.logger.info(f"[manual] FAIL: {e}")
    finally:
        if sim is not None:
            # PX4 warnings are informational for the manual smoke test, where success is the climb,
            # not a gate. The console is the log of the container this sim started.
            warn_lines = px4_warnings(sim.artifacts()["px4_log"])
            stats["px4_warnings"] = len(warn_lines)
            if warn_lines:
                na.logger.info(f"[manual] {len(warn_lines)} PX4 warning line(s) (informational):")
                for w in sorted(set(warn_lines))[:20]:
                    na.logger.info(f"  {w}")
            # Sim-side truth: the climb from the recorder history, the max altitude over the whole
            # flight. Measured here rather than on the happy path so a flight that timed out still
            # reports how far it actually got.
            if spawn_alt is not None:
                alts = np.array([s.altitude_m for s in sim.physics[sim.base_body].history()])
                climb = float(alts.max() - spawn_alt) if alts.size else 0.0
                stats.update(takeoff_confirmed=flight_ok, climb_m=round(climb, 2))
                na.logger.info(f"[manual] {'takeoff OK' if flight_ok else 'no takeoff'}, climbed {climb:.2f} m")
            stats["rtf"] = round(float(sim.results().get("rtf", 0.0)), 2)
            dump_run(sim, "px4_sitl_manual", stats=stats)  # evaluation artifacts: flown trajectory + stats

    na.logger.info(f"[manual] {json.dumps(stats)}  ->  {'FLIGHT OK' if flight_ok else 'FAIL'}")
    return 0 if flight_ok else 1


if __name__ == "__main__":
    sys.exit(main())
