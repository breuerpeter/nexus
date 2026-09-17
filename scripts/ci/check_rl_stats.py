"""Regression gate for the Astro Max RL example, the stats table in docs/examples/isaac-lab-rl.md.

Compares a fresh run's stats against ``nexus-rl/stats_baseline.json`` and exits non-zero if any
guarded stat regressed beyond tolerance, so training and deploy speed, and convergence via the deploy
tracking error, can't silently get worse. Runner-agnostic; the GPU CI job, scripts/ci/run_rl_example.sh, runs:

    uv run --project nexus-rl python nexus-rl/scripts/rsl_rl/train.py --stats_json train.json [other args]
    uv run --extra policy python nexus/examples/controllers/policy/goto/flight.py --stats-json deploy.json [other args]
    python scripts/ci/check_rl_stats.py --train-stats train.json --deploy-stats deploy.json

Each baseline entry is either a config value, an exact match such as ``num_envs``, or a rule dict with a
``value`` plus one guardrail: ``min_frac`` or ``max_frac``, which compare fresh to ``value·frac``, or
``min`` or ``max``, which are absolute. ``min_frac`` guards "higher is better" stats such as throughput;
``max_frac`` and ``max`` guard "lower is better" stats such as wall time or tracking error.
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-stats", help="JSON from train.py (NEWTON_RL_STATS_JSON)")
    ap.add_argument("--deploy-stats", help="JSON from deploy_rollout.py (--stats-json)")
    ap.add_argument("--baseline", default=str(BASELINE))
    args = ap.parse_args()

    base = json.loads(pathlib.Path(args.baseline).read_text())
    fresh = {}
    if args.train_stats:
        fresh["train"] = json.loads(pathlib.Path(args.train_stats).read_text())
    if args.deploy_stats:
        fresh["deploy"] = json.loads(pathlib.Path(args.deploy_stats).read_text())
    if not fresh:
        print("nothing to check: pass --train-stats and/or --deploy-stats", file=sys.stderr)
        return 2

    failed = []
    print(f"{'metric':<32}{'fresh':>14}{'guard':>16}  status")
    print("-" * 70)
    for group in ("train", "deploy"):
        if group not in fresh:
            continue
        for key, rule in base.get(group, {}).items():
            if key not in fresh[group]:
                print(f"{group + '.' + key:<32}{'MISSING':>14}{'':>16}  REGRESSED")
                failed.append(f"{group}.{key} (missing)")
                continue
            got = fresh[group][key]
            if not isinstance(rule, dict):  # config field: exact match, for example num_envs
                ok = got == rule
                print(f"{group + '.' + key:<32}{got!s:>14}{('== ' + str(rule)):>16}  {'ok' if ok else 'CHANGED'}")
                if not ok:
                    failed.append(f"{group}.{key} ({got} != {rule})")
                continue
            val = float(got)
            ok, guard = _evaluate(rule, val)
            print(f"{group + '.' + key:<32}{val:>14.4g}{guard:>16}  {'ok' if ok else 'REGRESSED'}")
            if not ok:
                failed.append(f"{group}.{key} ({val:.4g} {guard})")

    if failed:
        print(f"\nFAILED: {len(failed)} stat(s) regressed:\n  " + "\n  ".join(failed))
        return 1
    print("\nAll guarded stats within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
