"""mkdocs hook: render benchmark tables from the committed data under ``docs/data/``.

Three markers, replaced at build time. The CI legs refresh the data files via reviewed
data-refresh PRs, and the site never fetches anything at build or view time:

* ``<!-- benchmark-matrix -->``: the Real-Time Factor (RTF) matrix tables from
  ``docs/data/rtf_matrix.json``, produced by ``scripts/ci/benchmark_matrix.py``: one PX4
  4-waypoint mission per cell.
* ``<!-- benchmark-examples -->``: the cross-example table from
  ``docs/data/examples_bench.json``, produced by ``scripts/ci/evaluate_examples.py``.
* ``<!-- example-stats: <name> -->``: one example's measured metrics as a small table
  from the same data file; renders a "no CI data yet" note for examples not yet in the feed.

As with ``vehicle_previews.py``, this lives under ``docs/hooks`` as a build tool, not content.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_MATRIX = _REPO / "docs/data/rtf_matrix.json"
_EXAMPLES = _REPO / "docs/data/examples_bench.json"
_MARKER = re.compile(r"<!--\s*(?P<kind>benchmark-matrix|benchmark-examples|example-stats:\s*(?P<name>[\w.-]+))\s*-->")
_ENTRY = re.compile(r"(?P<metric>[\w.-]+)\[(?P<example>[\w.-]+)\]")


def _fmt(value, unit: str) -> str:
    if unit == "x realtime":
        return f"{value:.2f}×"
    if unit == "m":
        return f"{value:.3f} m"
    if unit == "deg":
        return f"{value:.1f}°"
    if unit == "s":
        return f"{value:.1f} s"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.4g}"
    return f"{int(value)}"


def _stamp(meta: dict) -> str:
    return (
        f"<small>Measured on **{meta.get('gpu', 'unknown')}** · {meta.get('recorded', '?')} · "
        f"commit `{meta.get('sha', '?')}`</small>"
    )


def _matrix_tables() -> str:
    if not _MATRIX.exists():
        return "<!-- benchmark-matrix: docs/data/rtf_matrix.json missing -->"
    data = json.loads(_MATRIX.read_text())
    meta, cells = data.get("_meta", {}), data.get("cells", [])
    out: list[str] = []

    def win(cell: dict) -> str:
        w = cell.get("rtf_win") or {}
        if not w:
            return ""
        return f"{w['min']:.2f} / {w['avg']:.2f} / {w['max']:.2f} (σ {w['std']:.2f})"

    physics = [c for c in cells if not c.get("renders") and c.get("rtf") is not None]
    if physics:
        out += ["**Physics only** (a vehicle with no RTX sensor, no renderer)", ""]
        out += ["| Vehicle | Scene | Device | Steady RTF | Window min / avg / max |", "|---|---|---|---|---|"]
        out += [
            f"| `{c['vehicle']}` | `{c['scene']}` | {c['device']} | **{c['rtf']:.2f}×** | {win(c)} |"
            for c in physics
        ]
        out.append("")

    rtx = [c for c in cells if c.get("renders") and c.get("rtf") is not None]
    if rtx:
        vehicles = list(dict.fromkeys(c["vehicle"] for c in rtx))
        scenes = list(dict.fromkeys(c["scene"] for c in rtx))
        by = {(c["vehicle"], c["scene"]): c for c in rtx}
        out += ["**RTX rendering** (the camera rendered by the Kit peer, GPU)", ""]
        out += ["| Vehicle | " + " | ".join(f"`{s}`" for s in scenes) + " |", "|---|" + "---|" * len(scenes)]
        for v in vehicles:
            row = [f"**{by[(v, s)]['rtf']:.2f}×**" if (v, s) in by else "" for s in scenes]
            out.append(f"| `{v}` | " + " | ".join(row) + " |")
        out.append("")
        if any(s == "cesium" for s in scenes):
            out += ["<small>`cesium` streams Google 3D Tiles over the network: its RTF varies with tile traffic.</small>", ""]
    else:
        out += ["*RTX cells pending the first scheduled `gpu-benchmark-matrix` run.*", ""]

    out.append(_stamp(meta) + f"<br><small>Mission: takeoff to {meta.get('alt_m', '?')} m, then waypoints `{meta.get('mission', '?')}` (x,y,z world offsets [m] from the takeoff point; +x is north).</small>")
    return "\n".join(out)


def _examples_data() -> tuple[dict, dict[str, list[dict]]]:
    data = json.loads(_EXAMPLES.read_text()) if _EXAMPLES.exists() else {}
    grouped: dict[str, list[dict]] = {}
    for e in data.get("entries", []):
        m = _ENTRY.fullmatch(e.get("name", ""))
        if m:
            grouped.setdefault(m["example"], []).append({**e, "metric": m["metric"]})
    return data.get("_meta", {}), grouped


def _examples_table() -> str:
    meta, grouped = _examples_data()
    if not grouped:
        return "<!-- benchmark-examples: docs/data/examples_bench.json missing/empty -->"
    out = ["| Example | Steady RTF | Pose APE rmse | Other measured metrics |", "|---|---|---|---|"]
    for example, entries in grouped.items():
        by = {e["metric"]: e for e in entries}
        rtf = f"**{by['rtf']['value']:.2f}×**" if "rtf" in by else ""
        ape = _fmt(by["ape_trans_rmse_m"]["value"], "m") if "ape_trans_rmse_m" in by else ""
        rest = ", ".join(
            f"{e['metric']} {_fmt(e['value'], e.get('unit', ''))}"
            for e in entries
            if e["metric"] not in ("rtf", "ape_trans_rmse_m")
        )
        out.append(f"| `{example}` | {rtf} | {ape} | {rest} |")
    out += ["", _stamp(meta)]
    return "\n".join(out)


def _example_stats(name: str) -> str:
    meta, grouped = _examples_data()
    entries = grouped.get(name)
    if not entries:
        return (
            f"*No CI-measured stats for `{name}` yet: they appear here after the first "
            f"`gpu-examples` run that covers it on the pinned runner.*"
        )
    out = ["| Metric | Value |", "|---|---|"]
    out += [f"| `{e['metric']}` | {_fmt(e['value'], e.get('unit', ''))} |" for e in entries]
    out += ["", _stamp(meta)]
    return "\n".join(out)


def on_page_markdown(markdown: str, *, page, config, files) -> str:
    if not _MARKER.search(markdown):
        return markdown

    def replace(match: re.Match) -> str:
        kind = match.group("kind")
        if kind == "benchmark-matrix":
            return _matrix_tables()
        if kind == "benchmark-examples":
            return _examples_table()
        return _example_stats(match.group("name"))

    return _MARKER.sub(replace, markdown)
