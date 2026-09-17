#!/usr/bin/env python3
"""The RTF benchmark matrix: ONE PX4 4-waypoint mission flown across {vehicle × scene × device ×
runtime} cells on the pinned runner hardware, producing the data the docs Benchmarking page renders.

Two tables:
    standalone: astro_max_base × empty × {cpu, cuda}
    isaacsim:   {base, fpv, fpv_lr1, fpv_flux} × {empty, cesium} × cuda

Per cell: run ``scripts/ci/benchmark_cell.py`` (plain Python for standalone, via ``nexus
script`` for isaacsim) and collect its ``--stats-json``. The cell is a whole flight, since its own sim
starts and stops PX4 and flies the mission, so this runner only sequences cells, times them out and
merges their numbers. Writes ``docs/data/rtf_matrix.json`` (consumed by ``docs/hooks/benchmarks.py``
at docs build) plus github-action-benchmark-style entries next to the per-cell artifacts.

cesium cells stream Google 3D Tiles (needs ``$CESIUM_ION_TOKEN``; RTF varies with network) and are
skipped with a note when the token is absent. isaacsim cells run with ``--benchmark`` so the
Kit-side recorders (GPU frametime, VRAM, RSS, RTF stability) collect too; their JSON lands in
``~/.cache/nexus/logs`` (bind-mounted out of the container) and is attached to the cell result.

    uv run python scripts/ci/benchmark_matrix.py [--only standalone|isaacsim] [--out docs/data/rtf_matrix.json]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

# The mission the cell flies, for the recorded metadata only, imported rather than restated so
# there is one definition of it. This file runs from scripts/ci, so the cell is an import away.
from benchmark_cell import ALT, MISSION

from nexus._src.containers import client

ROOT = pathlib.Path(__file__).resolve().parents[2]
VEHICLES = ("astro_max_base", "astro_max_fpv", "astro_max_fpv_lr1", "astro_max_fpv_flux")
SCENES = ("empty", "cesium")
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
    return "-".join(cell[k] for k in ("runtime", "device", "vehicle", "scene"))


def _kill_kit() -> None:
    """Force-remove any leftover isaacsim-runtime container: a hung cell's ``docker compose run``
    container survives killing the host client and would serve a stale sim on :4560 to the next
    cell, misattributing its numbers.
    """
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["standalone", "isaacsim"], default=None, help="run one table only")
    ap.add_argument("--out", default=str(ROOT / "docs" / "data" / "rtf_matrix.json"))
    ap.add_argument("--work", default=str(ROOT / ".eval-artifacts" / "matrix"), help="per-cell artifact dir")
    ap.add_argument("--budget", type=float, default=600.0, help="per-cell flight budget [sim s]")
    ap.add_argument("--cell-timeout", type=float, default=2400.0, help="per-cell wall leash [s]")
    args = ap.parse_args()
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    skipped: list[str] = []
    for cell in _cells(args.only):
        if cell["scene"] == "cesium" and not os.environ.get("CESIUM_ION_TOKEN"):
            print(f"skipping {_tag(cell)}: CESIUM_ION_TOKEN not set", flush=True)
            skipped.append(_tag(cell))
            continue
        results.append(_run_cell(cell, work, args.budget, args.cell_timeout))

    # Partial runs, with --only or cesium skipped without a token, must not clobber cells they didn't
    # fly: merge fresh cells over the committed matrix, keyed by the cell axes. The refresh PR
    # diff then shows exactly the re-measured cells.
    committed = ROOT / "docs" / "data" / "rtf_matrix.json"
    if committed.exists():

        def _key(c: dict):
            return tuple(c.get(k) for k in ("runtime", "device", "vehicle", "scene"))

        fresh_keys = {_key(c) for c in results}
        prior = json.loads(committed.read_text()).get("cells", [])
        results += [c for c in prior if _key(c) not in fresh_keys]

    sha = (
        os.environ.get("GITHUB_SHA")
        or subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=False
    ).stdout.strip()
    meta = {
        "sha": sha[:12],
        "recorded": time.strftime("%Y-%m-%d"),
        "gpu": gpu.splitlines()[0] if gpu else "unknown",
        "mission": "; ".join(f"{x:g},{y:g},{z:g}" for x, y, z in MISSION),
        "alt_m": ALT,
        "skipped": skipped,
    }
    out = pathlib.Path(args.out)
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

    bad = [_tag(c) for c in results if c.get("error") or not c.get("mission_ok")]
    if bad:
        print(f"FAILED cells: {', '.join(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
