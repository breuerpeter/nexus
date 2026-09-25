#!/usr/bin/env python3
"""The RTF benchmark matrix: ONE PX4 4-waypoint mission flown across {vehicle × scene × device ×
runtime} cells on the pinned runner hardware, producing the data the docs Benchmarking page renders.

Two tables:
    standalone: astro_max_base × empty × {cpu, cuda}
    isaacsim:   {base, fpv} × {empty, cesium} × cuda

Per cell: run ``scripts/ci/benchmark_cell.py`` (plain Python for standalone, via ``nexus
script`` for isaacsim) and collect its ``--stats-json``. The cell is a whole flight, since its own sim
starts and stops PX4 and flies the mission, so this runner only sequences cells, times them out and
merges their numbers. Writes ``docs/data/rtf_matrix.json`` (consumed by ``docs/hooks/benchmarks.py``
at docs build) plus github-action-benchmark-style entries next to the per-cell artifacts.

cesium cells stream Google 3D Tiles (needs ``$CESIUM_ION_TOKEN``; RTF varies with network) and are
skipped with a note when the token is absent. isaacsim cells run with ``--benchmark`` so the
Kit-side recorders (GPU frametime, VRAM, RSS, RTF stability) collect too; their JSON lands in
``~/.cache/nexus/logs`` (bind-mounted out of the container) and is attached to the cell result.

CI flies one cell per GPU box, since a measured flight cannot share a GPU or PX4's fixed ports, in
three steps of which only the middle one needs a GPU or a nexus install: ``--list`` prints the cells
to fly and the tags skipped as one JSON for the workflow matrix, ``--cell <tag>`` flies one cell and
leaves its result beside its stats, and ``--merge`` joins the results the boxes uploaded over the
committed matrix. With none of the three, the cells fly one after another on this machine.

    uv run python scripts/ci/benchmark_matrix.py [--only standalone|isaacsim] [--out docs/data/rtf_matrix.json]
    python3 scripts/ci/benchmark_matrix.py --list [--only …]
    uv run python scripts/ci/benchmark_matrix.py --cell <tag> [--work <dir>]
    python3 scripts/ci/benchmark_matrix.py --merge --plan '<the --list output>' [--work <dir>] [--out …]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
VEHICLES = ("astro_max_base", "astro_max_fpv")
SCENES = ("empty", "cesium")
AXES = ("runtime", "device", "vehicle", "scene")
KIT_LOGS = pathlib.Path.home() / ".cache" / "nexus" / "logs"  # bind-mounted out of the Kit container


def _cells(only: str | None) -> list[dict]:
    cells = []
    if only in (None, "standalone"):
        cells += [
            {"runtime": "standalone", "device": d, "vehicle": "astro_max_base", "scene": "empty"}
            for d in ("cpu", "cuda")
        ]
    if only in (None, "isaacsim"):
        cells += [{"runtime": "isaacsim", "device": "cuda", "vehicle": v, "scene": s} for v in VEHICLES for s in SCENES]
    return cells


def _tag(cell: dict) -> str:
    return "-".join(cell[k] for k in AXES)


def _plan(only: str | None) -> tuple[list[dict], list[str]]:
    """The cells to fly and the tags skipped: a cesium cell needs ``$CESIUM_ION_TOKEN``."""
    cells, skipped = [], []
    for cell in _cells(only):
        if cell["scene"] == "cesium" and not os.environ.get("CESIUM_ION_TOKEN"):
            skipped.append(_tag(cell))
        else:
            cells.append(cell)
    return cells, skipped


def _ok(result: dict) -> bool:
    return not result.get("error") and bool(result.get("mission_ok"))


def _kill_kit() -> None:
    """Force-remove any leftover isaacsim-runtime container: a hung cell's ``docker compose run``
    container survives killing the host client and would serve a stale sim on :4560 to the next
    cell, misattributing its numbers.
    """
    from nexus._src.containers import client  # here, not at the top: --list and --merge run with no nexus install

    for c in client().containers.list(filters={"label": "com.docker.compose.service=isaacsim"}):
        c.remove(force=True)


def _run_cell(cell: dict, work: pathlib.Path, budget_s: float, cell_timeout: float) -> dict:
    stats_json = work / f"{_tag(cell)}.json"
    entry = ["python"] if cell["runtime"] == "standalone" else ["nexus", "script"]
    # fmt: off
    cmd = [
        "uv", "run", *entry, "scripts/ci/benchmark_cell.py",
        "--vehicle", cell["vehicle"], "--scene", cell["scene"], "--device", cell["device"],
        "--log", "--stats-json", str(stats_json),
    ]
    # fmt: on
    env = {**os.environ, "NEWTON_CELL_FLY_S": str(budget_s)}
    kit_before: set[pathlib.Path] = set()
    if cell["runtime"] == "isaacsim":
        cmd.append("--benchmark")  # Kit-side recorders: the shared diagnostics flag rides argv into Kit
        kit_before = set(KIT_LOGS.glob("benchmark-*.json"))
    print(f"\n===== {_tag(cell)}: {' '.join(cmd)} =====", flush=True)

    result = dict(cell)
    if cell["runtime"] == "isaacsim":
        _kill_kit()
    stats_json.unlink(missing_ok=True)  # a stale per-tag file must never pass for fresh data
    # One wall leash for the whole cell, because the cell is now the whole flight: it builds PX4,
    # boots, where a cold Kit boot plus Warp compile measured ~17 min on a fresh CI box, reaches
    # lockstep, flies, and dumps its stats. A stale sim on :4560 needs no pre-check any more: it
    # surfaces as the cell's own bind failure.
    try:
        subprocess.run(cmd, cwd=ROOT, env=env, timeout=cell_timeout, check=False)
    except subprocess.TimeoutExpired:
        result["error"] = f"cell did not finish within {cell_timeout:.0f}s"
    finally:
        if cell["runtime"] == "isaacsim":
            _kill_kit()  # killing the host client doesn't stop the compose-run Kit container
    if stats_json.exists():
        result.update(json.loads(stats_json.read_text()))
    else:
        result.setdefault("error", "cell dumped no stats")
    if cell["runtime"] == "isaacsim":
        fresh = sorted(set(KIT_LOGS.glob("benchmark-*.json")) - kit_before)
        if fresh:
            result["kit"] = json.loads(fresh[-1].read_text())
    return result


def _box_meta() -> dict:
    """What the box that flew a cell knows for ``_meta``: its GPU, and the mission the cell flies."""
    # An import, not a copy, keeps one definition of the mission. It sits here, not at the top,
    # because it imports nexus and the merge runs on a box without it.
    from benchmark_cell import ALT, MISSION

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return {
        "gpu": gpu.splitlines()[0] if gpu else "unknown",
        "mission": "; ".join(f"{x:g},{y:g},{z:g}" for x, y, z in MISSION),
        "alt_m": ALT,
    }


def _finish(results: list[dict], skipped: list[str], box: dict, out: pathlib.Path, work: pathlib.Path) -> int:
    """Write the matrix and the bench entries from the fresh cells; 1 when any fresh cell failed."""
    fresh = list(results)
    # Partial runs, with --only or cesium skipped without a token, must not clobber cells they didn't
    # fly: merge fresh cells over the committed matrix, keyed by the cell axes. The refresh PR
    # diff then shows exactly the re-measured cells.
    committed = ROOT / "docs" / "data" / "rtf_matrix.json"
    if committed.exists():

        def _key(c: dict):
            return tuple(c.get(k) for k in AXES)

        fresh_keys = {_key(c) for c in results}
        prior = json.loads(committed.read_text()).get("cells", [])
        results = results + [c for c in prior if _key(c) not in fresh_keys]

    sha = (
        os.environ.get("GITHUB_SHA")
        or subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    meta = {"sha": sha[:12], "recorded": time.strftime("%Y-%m-%d"), **box, "skipped": skipped}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"_meta": meta, "cells": results}, indent=2) + "\n")
    print(f"\nwrote {out} ({len(results)} cells, {len(skipped)} skipped)", flush=True)

    bench = [
        {
            "name": f"rtf[{c['vehicle']}|{c['runtime']}|{c['device']}|{c['scene']}]",
            "unit": "x realtime",
            "value": c["rtf"],
            "biggerIsBetter": True,
        }
        for c in results
        if c.get("rtf") is not None
    ]
    (work / "benchmark.json").write_text(json.dumps(bench, indent=2))

    bad = [_tag(c) for c in fresh if not _ok(c)]
    if bad:
        print(f"FAILED cells: {', '.join(bad)}", flush=True)
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["standalone", "isaacsim"], default=None, help="run one table only")
    ap.add_argument("--list", action="store_true", help="print the cells to fly and the tags skipped as JSON")
    ap.add_argument("--cell", default=None, metavar="TAG", help="fly one cell and leave its result in --work")
    ap.add_argument("--merge", action="store_true", help="join the cell results in --work into the matrix")
    ap.add_argument("--plan", default=None, help="with --merge: the --list output, for the skipped cells")
    ap.add_argument("--out", default=str(ROOT / "docs" / "data" / "rtf_matrix.json"))
    ap.add_argument("--work", default=str(ROOT / ".eval-artifacts" / "matrix"), help="per-cell artifact dir")
    ap.add_argument("--budget", type=float, default=600.0, help="per-cell flight budget [sim s]")
    ap.add_argument("--cell-timeout", type=float, default=2400.0, help="per-cell wall leash [s]")
    args = ap.parse_args()

    if args.list:
        cells, skipped = _plan(args.only)
        print(json.dumps({"cells": [{**c, "tag": _tag(c)} for c in cells], "skipped": skipped}))
        return 0

    work = pathlib.Path(args.work).resolve()  # the cell runs from ROOT, whatever this process's cwd
    work.mkdir(parents=True, exist_ok=True)
    out = pathlib.Path(args.out)

    if args.cell:
        cell = next((c for c in _cells(None) if _tag(c) == args.cell), None)
        if cell is None:
            ap.error(f"unknown cell {args.cell!r}; the tags: {', '.join(_tag(c) for c in _cells(None))}")
        result = _run_cell(cell, work, args.budget, args.cell_timeout)
        # The result and what only this box knows, for --merge on a box with no GPU.
        (work / f"{args.cell}.cell.json").write_text(json.dumps({"result": result, **_box_meta()}, indent=2) + "\n")
        if not _ok(result):
            print(f"FAILED cell: {args.cell}: {result.get('error') or 'mission not completed'}", flush=True)
            return 1
        return 0

    if args.merge:
        parts = sorted(work.glob("*.cell.json"))
        if not parts:
            print(f"no cell result under {work}", flush=True)
            return 1
        loaded = [json.loads(p.read_text()) for p in parts]
        skipped = json.loads(args.plan)["skipped"] if args.plan else []
        box = {k: loaded[0][k] for k in ("gpu", "mission", "alt_m")}
        return _finish([p["result"] for p in loaded], skipped, box, out, work)

    cells, skipped = _plan(args.only)
    for tag in skipped:
        print(f"skipping {tag}: CESIUM_ION_TOKEN not set", flush=True)
    results = [_run_cell(cell, work, args.budget, args.cell_timeout) for cell in cells]
    return _finish(results, skipped, _box_meta(), out, work)


if __name__ == "__main__":
    sys.exit(main())
