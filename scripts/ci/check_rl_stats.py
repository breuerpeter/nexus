"""Regression gate for the Astro Max RL example's training half, the stats in docs/examples/isaac-lab-rl.md.

Compares a fresh training run's stats, ``train.py --stats_json``, against ``nexus-rl/stats_baseline.json``
and exits non-zero if a guarded stat regressed, so a change that breaks learning can't land silently.
Wall-clock stats, ``steps_per_sec`` and ``train_seconds``, print as monitored and never gate: no throughput
baseline holds on a shared runner. The deploy-side gates live with the examples evaluation harness,
scripts/ci/evaluate_examples.py. Runner-agnostic; the GPU CI job, scripts/ci/run_rl_example.sh, runs:

    uv run --project nexus-rl python nexus-rl/scripts/rsl_rl/train.py --stats_json train.json [other args]
    python scripts/ci/check_rl_stats.py --train-stats train.json

Each baseline entry is either a config value, an exact match such as ``num_envs``, or a rule dict with one
guardrail: ``min`` or ``max``, absolute, or ``min_frac`` or ``max_frac``, which compare fresh to
``value·frac``. A numeric stat the baseline doesn't name prints as monitored.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASELINE = ROOT / "nexus-rl" / "stats_baseline.json"


def _evaluate(rule, fresh: float):
    """Return a pass flag and the threshold string for a numeric rule dict."""
    v = rule.get("value")
    if "min_frac" in rule:
        lo = v * rule["min_frac"]
        return fresh >= lo, f">= {lo:.4g}"
    if "max_frac" in rule:
        hi = v * rule["max_frac"]
        return fresh <= hi, f"<= {hi:.4g}"
    if "min" in rule:
        return fresh >= rule["min"], f">= {rule['min']}"
    if "max" in rule:
        return fresh <= rule["max"], f"<= {rule['max']}"
    return True, "(no guard)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-stats", required=True, help="JSON from train.py --stats_json")
    ap.add_argument("--baseline", default=str(BASELINE))
    args = ap.parse_args()

    rules = json.loads(pathlib.Path(args.baseline).read_text())["train"]
    fresh = json.loads(pathlib.Path(args.train_stats).read_text())

    failed = []
    print(f"{'metric':<32}{'fresh':>14}{'guard':>16}  status")
    print("-" * 70)
    for key, rule in rules.items():
        name = f"train.{key}"
        if key not in fresh:
            print(f"{name:<32}{'MISSING':>14}{'':>16}  REGRESSED")
            failed.append(f"{name} (missing)")
            continue
        got = fresh[key]
        if not isinstance(rule, dict):  # config field: exact match, for example num_envs
            ok = got == rule
            print(f"{name:<32}{got!s:>14}{('== ' + str(rule)):>16}  {'ok' if ok else 'CHANGED'}")
            if not ok:
                failed.append(f"{name} ({got} != {rule})")
            continue
        val = float(got)
        ok, guard = _evaluate(rule, val)
        print(f"{name:<32}{val:>14.4g}{guard:>16}  {'ok' if ok else 'REGRESSED'}")
        if not ok:
            failed.append(f"{name} ({val:.4g} {guard})")
    for key in sorted(set(fresh) - set(rules)):  # monitored but not gated: the wall clock, the run's shape
        val = fresh[key]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        print(f"{'train.' + key:<32}{val:>14.4g}{'(monitored)':>16}  -")

    if failed:
        print(f"\nFAILED: {len(failed)} stat(s) regressed:\n  " + "\n  ".join(failed))
        return 1
    print("\nAll guarded stats within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
