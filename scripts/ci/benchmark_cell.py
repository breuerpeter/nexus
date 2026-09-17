#!/usr/bin/env python3
# Acronyms: Forward Left Up (FLU).
r"""One benchmark-matrix cell: a whole PX4 flight in one process, parameterized by the standard CLI
flags. The SAME file runs on both runtimes (the matrix runner picks the entry):

    standalone:  uv run python scripts/ci/benchmark_cell.py --vehicle astro_max_base --device cpu \\
                     --log --stats-json .eval-artifacts/matrix/cell.json
    isaacsim:    uv run nexus script scripts/ci/benchmark_cell.py --vehicle astro_max_fpv \\
                     --scene cesium --log --stats-json …

(``nexus script`` re-runs the file inside the booted Kit container; ``na.Sim`` detects the
booted Kit and builds through the isaacsim glue: the identical script, one code path per cell.)

The sim owns PX4: it builds it, serves HIL on :4560, starts the container, and kills it on the way
out. So the cell flies THE 4-waypoint mission itself and no external peer's death can end its run.
The stats (``mission_ok``, steady ``rtf`` + the profiler's whole-run window spread ``rtf_win``) land
in ``--stats-json``, which the matrix runner collects.
"""

from __future__ import annotations

import os
import sys

import nexus as na

READY_S = float(os.environ.get("NEWTON_CELL_READY_S", "300"))  # PX4-lockstep wait budget
FLY_S = float(os.environ.get("NEWTON_CELL_FLY_S", "600"))  # arm + climb budget [sim s]

# The benchmark mission: 4 waypoints in world axes, Newton FLU, Z-up, relative to the takeoff
# point. North is +x, so east is -y. These are the same four points the matrix has always flown;
# tests/operator/test_px4_offboard.py pins that equivalence, because the baselines are only
# comparable across runs if the flown path is the same.
MISSION = ((20.0, 0.0, 5.0), (20.0, -20.0, 8.0), (0.0, -20.0, 5.0), (0.0, 0.0, 5.0))
ALT = 5.0  # takeoff altitude [m]; the mission flies relative to the takeoff point
WP_ARRIVE_M = 2.0  # 3D arrival radius per waypoint [m]
WP_TIMEOUT_S = 90.0  # per-waypoint budget [sim s]


def _fly_mission(sim) -> bool:
    """Fly the benchmark mission over the operator link on :14540 and report whether it completed.

    A plain script against ``sim.operator``: take off, then each waypoint in turn, waiting on PX4's
    own telemetry for arrival. The waits are ``sim.wait_until``, so the budgets are in sim seconds
    and a slow cell isn't a failed one.

    Args:
        sim: The running :class:`~nexus.Sim`, already at lockstep.

    Returns:
        ``True`` if the takeoff and every waypoint completed.
    """
    try:
        op = sim.operator  # built lazily on first access; must come after start()
        op.takeoff(ALT)
        sim.wait_until(op.at_target, sim_timeout=FLY_S)
        for i, wp in enumerate(MISSION, 1):
            op.goto(wp)
            sim.wait_until(op.at_target, sim_timeout=WP_TIMEOUT_S)
            print(f"[bench] wp {i}/{len(MISSION)} reached {wp}", flush=True)
        return True
    except (TimeoutError, RuntimeError, OSError) as e:
        print(f"[bench] mission FAILED: {e}", flush=True)
        return False


def main() -> int:
    args = na.sim_argparser(description="benchmark-matrix cell (one whole PX4 flight)").parse_args()
    with na.Sim.from_args(args) as sim:
        sim.start(timeout=READY_S)
        mission_ok = _fly_mission(sim)
    # Leaving the `with` stops the sim and kills PX4, so the run is over and the stats are final.
    results = sim.results()
    stats = {
        "mission_ok": mission_ok,
        "rtf": results.get("rtf"),
        "full_rtf": results.get("full_rtf"),
        "control_steps": results.get("control_steps"),
        "rtf_win": (results.get("profile") or {}).get("rtf_win"),
    }
    na.save_run_artifacts(sim, args, stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
