#!/usr/bin/env python3
# Acronyms: Forward Left Up (FLU).
r"""One benchmark-matrix cell: a whole PX4 flight in one process, parameterized by the standard CLI
flags. Every cell runs the same way; a vehicle with a camera renders it in the Kit peer, which the
sim starts itself:

    uv run python scripts/ci/benchmark_cell.py --vehicle astro_max_fpv --scene cesium --device cuda \\
        --log --stats-json .eval-artifacts/matrix/cell.json

The sim owns PX4: it builds it, serves HIL on :4560, starts the container, and kills it on the way
out. So the cell flies its mission itself and no external peer's death can end its run: the shared
4-waypoint square by default, or with ``--mission inspection`` the ``powerline`` scene's flight
along its line, which PX4 flies as an uploaded mission.
The stats (``mission_ok``, steady ``rtf`` + the profiler's whole-run window spread ``rtf_win``) land
in ``--stats-json``, which the matrix runner collects.
"""

from __future__ import annotations

import math
import os
import sys

import nexus as na
from nexus._src.operator.qgc_plan import FRAME_GLOBAL_RELATIVE_ALT, NAV_TAKEOFF, NAV_WAYPOINT, MissionItem

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

# The inspection of the ``powerline`` scene, in the same world axes, from its registry ``start``
# beside a pole: climb to 2 m over the line's top conductor, fly straight over the line 20 m up the
# drop at 2 m/s, and land beside the far pole on the high ground. The line runs along -x at y = 2;
# the near pole's base is at z = 0 and the far one's at z = 3, the top conductor 5.69 m over each.
INSPECTION_START = (0.0, 2.0, 7.69)
INSPECTION_END = (-20.0, 2.0, 10.69)
INSPECTION_LAND = (-20.0, 0.0)
INSPECTION_SPEED = 2.0  # [m/s]
# Where the world origin sits on the globe for the inspection: the mission's items are geodetic, so
# the cell pins the origin rather than read it back from the run.
INSPECTION_GEO = (47.397742, 8.545594)
NAV_LAND = 21
DO_CHANGE_SPEED = 178
FRAME_MISSION = 2  # the frame PX4 reads a DO_ command in; it refuses one in a global frame
_R_EARTH = 6378137.0
INSPECTION_S = 300.0  # budget [sim s] for the whole inspection, upload to landing


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


def _inspection_items() -> list[MissionItem]:
    """The inspection as PX4 mission items, geodetic around ``INSPECTION_GEO``, altitudes over home."""
    lat0, lon0 = INSPECTION_GEO

    def fix(x: float, y: float) -> tuple[int, int]:
        # World +x is north and +y is west, on a spherical earth's local tangent plane.
        lat = lat0 + math.degrees(x / _R_EARTH)
        lon = lon0 + math.degrees(-y / (_R_EARTH * math.cos(math.radians(lat0))))
        return round(lat * 1e7), round(lon * 1e7)

    nan = float("nan")
    steps = [
        (NAV_TAKEOFF, (0.0, 0.0, 0.0, nan), fix(0.0, 0.0), INSPECTION_START[2]),
        (DO_CHANGE_SPEED, (1.0, INSPECTION_SPEED, -1.0, 0.0), None, 0.0),
        (NAV_WAYPOINT, (0.0, 0.0, 0.0, nan), fix(*INSPECTION_START[:2]), INSPECTION_START[2]),
        (NAV_WAYPOINT, (0.0, 0.0, 0.0, nan), fix(*INSPECTION_END[:2]), INSPECTION_END[2]),
        (NAV_LAND, (0.0, 0.0, 0.0, nan), fix(*INSPECTION_LAND), 0.0),
    ]
    return [
        MissionItem(seq, FRAME_MISSION, cmd, True, params, 0, 0, z)
        if at is None
        else MissionItem(seq, FRAME_GLOBAL_RELATIVE_ALT, cmd, True, params, *at, z)
        for seq, (cmd, params, at, z) in enumerate(steps)
    ]


def _fly_inspection(sim) -> bool:
    """Upload the inspection, fly it in Mission mode and report whether it landed at its end."""
    try:
        op = sim.operator
        op.upload_mission(_inspection_items())
        sim.wait_until(op.mission_uploaded, sim_timeout=60.0)
        op.start_mission()
        sim.wait_until(op.mission_complete, sim_timeout=INSPECTION_S)
        print("[bench] inspection: flew the line", flush=True)
        sim.wait_until(lambda: op.landed_state() == "ON_GROUND", sim_timeout=INSPECTION_S)
        print("[bench] inspection: landed", flush=True)
        return True
    except (TimeoutError, RuntimeError, OSError) as e:
        print(f"[bench] mission FAILED: {e}", flush=True)
        return False


def main() -> int:
    parser = na.sim_argparser(description="benchmark-matrix cell (one whole PX4 flight)")
    parser.add_argument("--mission", choices=("square", "inspection"), default="square", help="the flight to fly")
    args = parser.parse_args()
    if args.mission == "inspection" and args.geo is None:
        args.geo = f"{INSPECTION_GEO[0]},{INSPECTION_GEO[1]}"
    with na.Sim.from_args(args) as sim:
        sim.start(timeout=READY_S)
        mission_ok = _fly_inspection(sim) if args.mission == "inspection" else _fly_mission(sim)
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
