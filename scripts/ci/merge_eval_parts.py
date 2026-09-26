#!/usr/bin/env python3
"""Join the per-example parts of a fanned-out gpu-examples run into one ``.eval-artifacts``.

Each GPU box flies one example and uploads its ``.eval-artifacts`` as ``examples-eval-<example>``.
Downloaded one folder per part, they join here into the one ``examples-eval`` tree a reader and the
data-refresh PR expect: every recording and score file copied over, and the files every part writes
merged by example, ``benchmark.json`` and ``examples_bench.json`` by entry name, and
``examples_baselines.json``, present after ``--update-baselines``, each example's block from the part
that flew it. Standard library only: the merge job runs on a plain runner.

    python3 scripts/ci/merge_eval_parts.py <parts dir> <out dir>
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys

PREFIX = "examples-eval-"
MERGED = ("benchmark.json", "examples_bench.json", "examples_baselines.json")


def merge_entries(prior: list, fresh: list[dict]) -> list[dict]:
    """Merge ``fresh`` bench entries over ``prior`` by entry name, fresh winning."""
    merged = {e["name"]: e for e in prior if isinstance(e, dict) and "name" in e}
    merged.update({e["name"]: e for e in fresh})
    return list(merged.values())


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    parts_dir, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    parts = sorted(p for p in parts_dir.iterdir() if p.is_dir() and p.name.startswith(PREFIX))
    if not parts:
        print(f"no {PREFIX}* part under {parts_dir}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    bench: list[dict] = []
    feed: dict = {"_meta": {}, "entries": []}
    baselines: dict = {}
    for part in parts:
        example = part.name[len(PREFIX) :]
        for src in sorted(p for p in part.rglob("*") if p.is_file()):
            rel = src.relative_to(part)
            if src.parent != part or rel.name not in MERGED:
                dst = out / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                continue
            data = json.loads(src.read_text())
            if rel.name == "benchmark.json":
                bench = merge_entries(bench, data)
            elif rel.name == "examples_bench.json":
                feed["_meta"] = data.get("_meta", feed["_meta"])
                feed["entries"] = merge_entries(feed["entries"], data.get("entries", []))
            else:
                # Every part carries the whole file with its own example refreshed, so the first
                # part seeds the merge and each part contributes the block it flew.
                if not baselines:
                    baselines = dict(data)
                baselines[example] = data[example]
        print(f"{example}: joined", flush=True)

    (out / "benchmark.json").write_text(json.dumps(bench, indent=2))
    (out / "examples_bench.json").write_text(json.dumps(feed, indent=2) + "\n")
    if baselines:
        (out / "examples_baselines.json").write_text(json.dumps(baselines, indent=2) + "\n")
    print(f"joined {len(parts)} parts into {out}: {len(bench)} bench entries", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
