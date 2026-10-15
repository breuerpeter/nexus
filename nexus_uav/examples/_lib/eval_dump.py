"""Standard per-run evaluation artifacts: the examples' half of the CI evaluation.

Every example dumps its flown trajectory, its reference when one exists, and its run stats to a
small ``<name>.npz`` + ``<name>.json`` pair via :func:`dump_run`, or via :func:`dump_stats` for a
non-flight example. The files are plain numpy/JSON. The evaluation tool that *scores* them, evo,
which is heavy and General Public License (GPL) licensed, lives only in the CI harness,
``scripts/ci/evaluate_examples.py``, which reads these artifacts; the examples themselves stay
dependency-light.

Conventions, which the harness relies on:

- ``est_t`` [s] / ``est_pos`` (N,3) / ``est_quat_xyzw`` (N,4): the flown base-body trajectory
  from the recorder, in the world frame, Z-up Forward Left Up (FLU), ground truth.
- Tracking reference, from acados: ``ref_t`` / ``ref_pos`` / ``ref_quat_wxyz`` sampled at the
  recorded timestamps that fall inside the reference's active window. The pairs are time-aligned by
  construction, so the harness computes the Absolute Pose Error (APE) directly, with no association
  step.
- Waypoint mission: ``waypoints`` (K,3) + ``arrival_t``, with as many entries as goals reached,
  from ``MissionGuidance.arrival_times``; the harness builds the piecewise-linear position
  reference.
- ``<name>.json``: ``{"name", "stats", "results", "rrd"}``, the example's stats dict, the
  loop's ``run_stats`` with the steady ``rtf`` and so on, and the ``.rrd`` path, ``null``
  unless the run recorded one.
- An example the harness times reports :meth:`RunClock.stats`: its ``wall_s``, the ``compile_s`` of it Warp
  spent compiling kernels, and ``warm_wall_s``, the rest. A time gate reads ``warm_wall_s``, so a cold kernel
  cache's one-time compiles don't fail it, as a benchmark warms its just-in-time compiler before it times.

Output dir: ``$NEXUS_EVAL_OUT``, which the CI harness sets, else ``~/.cache/nexus/eval``.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import warp as wp

# How Warp reports a module it compiled, through its logger: ``Module <name> <hash> load on device '<device>'
# took <ms> ms (compiled)``. A module it loaded from its cache ends in ``(cached)``.
_COMPILED = re.compile(r" took ([0-9.]+) ms\s+\(compiled\)")


class RunClock:
    """An example run's wall time, and the part of it Warp spent compiling kernels.

    Inside its ``with`` block the clock is Warp's logger: it reads each report of a module Warp compiled, and
    passes every report on to the logger it replaced, through :class:`warp.ScopedLogger`. When a report's
    format changes, the clock counts no compile, so a time gate goes back to the whole wall time: it fails
    on a cold cache, never passes a slow run.
    """

    def __init__(self):
        self._compile_s = 0.0
        self._start = 0.0
        self._replaced = wp.get_logger()
        self._scope = wp.ScopedLogger(self)

    def __enter__(self) -> RunClock:
        self._replaced = wp.get_logger()
        self._scope.__enter__()
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._scope.__exit__(exc_type, exc_value, traceback)

    def stats(self) -> dict[str, float]:
        """The run so far [s]: its ``wall_s``, the ``compile_s`` of it Warp spent compiling kernels, and
        ``warm_wall_s``, the rest, which a time gate reads.
        """
        wall = time.perf_counter() - self._start
        return {
            "wall_s": round(wall, 1),
            "compile_s": round(self._compile_s, 1),
            "warm_wall_s": round(wall - self._compile_s, 1),
        }

    # Warp's logger protocol: each message passes on to the logger the clock replaced.
    def debug(self, message: str) -> None:
        self._replaced.debug(message)

    def info(self, message: str) -> None:
        if hit := _COMPILED.search(message):
            self._compile_s += float(hit.group(1)) / 1000.0
        self._replaced.info(message)

    def warning(self, message: str, category=None, stacklevel: int = 1) -> None:
        self._replaced.warning(message, category, stacklevel + 1)

    def error(self, message: str) -> None:
        self._replaced.error(message)


def _out_dir(out_dir: str | Path | None) -> Path:
    d = Path(out_dir or os.environ.get("NEXUS_EVAL_OUT") or Path.home() / ".cache" / "newton" / "eval")
    d.mkdir(parents=True, exist_ok=True)
    return d


def dump_stats(name: str, stats: dict, *, results: dict | None = None, out_dir: str | Path | None = None) -> Path:
    """Write the JSON half only, for an example such as ``mass_recovery`` that has no flight trajectory.

    Args:
        name: The example's launcher name, as in ``python -m nexus_uav.examples <name>``, and the
            artifact stem.
        stats: The example's metrics dict, of plain JSON-serializable values.
        results: Optional run-stats mapping, shaped as ``sim.results()`` with ``rtf`` and so on.
        out_dir: Override the output directory; the default is ``$NEXUS_EVAL_OUT``, else the
            newton cache.

    Returns:
        Path: The written ``<name>.json``.
    """
    d = _out_dir(out_dir)
    path = d / f"{name}.json"
    path.write_text(json.dumps({"name": name, "stats": stats, "results": results or {}, "rrd": None}, indent=2))
    return path


def dump_run(
    sim,
    name: str,
    *,
    stats: dict | None = None,
    reference=None,
    reference_t0: float | None = None,
    waypoints=None,
    arrival_times=None,
    out_dir: str | Path | None = None,
) -> Path:
    """Dump a flown run's evaluation artifacts, ``<name>.npz`` + ``<name>.json``.

    Call after the ``Sim`` context exits; the recorder history and ``sim.results()`` persist on the
    handle. Pass exactly one reference form, or none:

    - ``reference`` + ``reference_t0``: a continuous planned reference, an object such as
      ``MinSnapReference`` or ``FlatnessReference`` that exposes
      ``flat_state_at(t) -> (pos, quat_wxyz, ...)`` and ``duration``, anchored at sim time
      ``reference_t0``, which is ``TrackingGuidance.reference_started_at``. Sampled at the
      recorded timestamps.
    - ``waypoints`` + ``arrival_times``: a goal-sequencing mission, from
      ``MissionGuidance.arrival_times``; the harness interpolates the position reference.

    Args:
        sim: The ``nx.Sim`` handle after its context exited.
        name: The example's launcher name, and the artifact stem.
        stats: The example's metrics dict, of plain JSON-serializable values.
        reference: Continuous reference object, for tracking controllers, or ``None``.
        reference_t0: Sim time [s] at which the controller received the reference.
        waypoints: The mission waypoints ``[(x, y, z), ...]``, or ``None``.
        arrival_times: Sim times [s] at which the vehicle reached each waypoint; can be shorter
            than ``waypoints``.
        out_dir: Override the output directory; the default is ``$NEXUS_EVAL_OUT``, else the
            newton cache.

    Returns:
        Path: The written ``<name>.json``; the ``.npz`` sits beside it.
    """
    d = _out_dir(out_dir)
    states = sim[f"vehicle/body/{sim.base_body}"].history()
    arrays: dict[str, np.ndarray] = {
        "est_t": np.asarray(states["t"], dtype=np.float64),
        "est_pos": np.asarray(states["position"], dtype=np.float64),
        "est_quat_xyzw": np.asarray(states["orientation"], dtype=np.float64),
    }
    if reference is not None:
        if reference_t0 is None:
            raise ValueError("dump_run(reference=...) needs reference_t0 (TrackingGuidance.reference_started_at)")
        dur = float(reference.duration)
        t = arrays["est_t"]
        sel = t >= float(reference_t0)
        ref_t = t[sel]
        samples = [reference.flat_state_at(min(float(ti) - float(reference_t0), dur)) for ti in ref_t]
        arrays["ref_t"] = ref_t
        arrays["ref_pos"] = np.array([s[0] for s in samples], dtype=np.float64)
        arrays["ref_quat_wxyz"] = np.array([s[1] for s in samples], dtype=np.float64)
    if waypoints is not None:
        arrays["waypoints"] = np.asarray(waypoints, dtype=np.float64)
        arrays["arrival_t"] = np.asarray(list(arrival_times or []), dtype=np.float64)
    np.savez_compressed(d / f"{name}.npz", **arrays)
    path = d / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "name": name,
                "stats": stats or {},
                "results": {k: v for k, v in sim.results().items() if isinstance(v, (int, float, str, bool))},
                "rrd": (sim.artifacts() or {}).get("rrd"),
            },
            indent=2,
        )
    )
    return path
