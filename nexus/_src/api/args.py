"""The shared command-line arg surface, import-light BY DESIGN and stdlib only.

The ``nexus`` command-line tool parses args before any runtime boots, and inside the Kit container
``warp``/``newton`` only become importable after ``SimulationApp`` starts, when the ``isaacsim.pip.newton``
extension adds them to ``sys.path``. So the parser must not live with ``Sim``, whose module pulls the whole
physics stack: this module imports nothing but the stdlib. ``Sim``-side consumers get these via
``nexus._src.api`` / ``na.sim_argparser`` as before.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from nexus._src.diagnostics import add_diagnostics_args  # stdlib-only, as is this module

if TYPE_CHECKING:
    from .sim import Sim


def sim_argparser(description: str | None = None) -> argparse.ArgumentParser:
    """The shared argument parser for ``Sim``-driven scripts and the ``nexus`` command-line tool.

    Carries the common flags :meth:`Sim.from_args` reads, ``--vehicle`` / ``--control`` / ``--device``
    / ``--scene`` / ``--max-steps`` / ``--log`` / ``--view``, so the command-line tool and any Sim-driven
    script share one arg surface. The bundled examples are zero-arg by design, and their configuration
    lives in the script; this parser serves the ``nexus`` command-line tool.
    """
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--vehicle", default=None,
        help="registry vehicle NAME (e.g. astro_max_fpv) or a local .usd path; omit to take the\n"
        "registry default",
    )  # fmt: skip
    p.add_argument(
        "--registry", default=None,
        help="path to the catalog to fly; omit to take the nearest nexus.registry.yaml at or above the\n"
        "working directory, else the one bundled in the wheel",
    )  # fmt: skip
    p.add_argument("--control", default="px4-sitl", help="control kind (px4-sitl)")
    p.add_argument(
        "--device", default="auto", choices=["cpu", "cuda", "auto"], help="compute device (cpu=deterministic)"
    )
    p.add_argument(
        "--scene", default=None,
        help="registry scene NAME (e.g. 'slalom') or a local scene .usdz path (a converted\n"
        "mesh/splat, flown as the visual world)",
    )  # fmt: skip
    p.add_argument(
        "--geo", default=None, metavar="LAT,LON[,ALT]",
        help="override the scene's geodetic origin, e.g. 37.79,-122.40 (cesium: where the globe streams)",
    )  # fmt: skip
    p.add_argument(
        "--solver", default=None, choices=["mujoco", "semi_implicit", "featherstone"],
        help="physics integrator (default: registry default, mujoco)",
    )  # fmt: skip
    p.add_argument("--max-steps", type=int, default=None, help="cap the run at N control steps")
    p.add_argument(
        "--rtf", type=float, default=0.0,
        help="real-time-factor throttle: 0 = unthrottled (default); 1.0 = pace to wall-clock for interactive flying",
    )  # fmt: skip
    p.add_argument("--log", action="store_true", help="write the Rerun .rrd to disk")
    p.add_argument("--view", action="store_true", help="serve the live Rerun viewer on :9876")
    p.add_argument(
        "--debug", action="store_true",
        help="axes-only scene: log each body's coordinate-frame triad instead of its mesh (much smaller .rrd)",
    )  # fmt: skip
    p.add_argument("--stats-json", default=None, help="write the run's stats dict to this JSON path")
    p.add_argument("--rrd-out", default=None, help="copy the run's .rrd here (e.g. for a docs page)")
    add_diagnostics_args(p)  # the shared --profile/--trace/--benchmark; Sim.from_args reads them
    return p


def save_run_artifacts(sim: Sim, args: argparse.Namespace, stats: dict | None = None) -> None:
    """Honor the shared ``--stats-json`` / ``--rrd-out`` flags after a run.

    Writes ``stats``, if given, to ``args.stats_json`` and copies the run's ``.rrd`` to ``args.rrd_out``.
    Call it after the ``Sim`` context exits: the close flushes the ``.rrd``, and ``sim.artifacts()``
    then reports its path. A no-op for flags the caller didn't set.
    """
    import json
    import shutil

    if stats is not None and getattr(args, "stats_json", None):
        with open(args.stats_json, "w") as f:
            json.dump(stats, f, indent=2)
    dest = getattr(args, "rrd_out", None)
    if dest:
        rrd = (sim.artifacts() or {}).get("rrd")
        if rrd:
            shutil.copy2(rrd, dest)
