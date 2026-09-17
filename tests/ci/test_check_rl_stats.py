"""The RL-example regression gate, scripts/ci/check_rl_stats.py, must pass on healthy stats and fail
when a guarded stat regresses. It runs against the committed baseline so CPU CI tests the gate itself,
while the full GPU pipeline that produces fresh stats runs on the gpu runner. The deploy-side
gates moved to the examples evaluation harness, scripts/ci/evaluate_examples.py, see
tests/ci/test_evaluate_examples.py; this baseline now guards training only.
"""

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CHECK = ROOT / "scripts" / "ci" / "check_rl_stats.py"
BASELINE = json.loads((ROOT / "nexus-rl" / "stats_baseline.json").read_text())


def _run(tmp_path, train):
    tj = tmp_path / "train.json"
    tj.write_text(json.dumps(train))
    return subprocess.run(
        [sys.executable, str(CHECK), "--train-stats", str(tj)],
        capture_output=True,
        text=True,
        check=False,
    )


# Healthy stats: exactly the baseline values, so this passes by construction whatever the pinned
# hardware numbers are. The baseline is re-recorded per CI runner, so no literals here.
_HEALTHY_TRAIN = {
    "num_envs": BASELINE["train"]["num_envs"],
    "steps_per_sec": BASELINE["train"]["steps_per_sec"]["value"],
    "train_seconds": BASELINE["train"]["train_seconds"]["value"],
}


def test_baseline_is_well_formed():
    assert BASELINE["train"]["num_envs"] == 2048
    assert "min_frac" in BASELINE["train"]["steps_per_sec"]
    assert "deploy" not in BASELINE  # deploy gates live in scripts/ci/examples_baselines.json now


def test_healthy_stats_pass(tmp_path):
    r = _run(tmp_path, _HEALTHY_TRAIN)
    assert r.returncode == 0, r.stdout + r.stderr


def test_throughput_regression_fails(tmp_path):
    rule = BASELINE["train"]["steps_per_sec"]
    bad = {**_HEALTHY_TRAIN, "steps_per_sec": rule["value"] * rule["min_frac"] * 0.9}  # under the floor
    r = _run(tmp_path, bad)
    assert r.returncode == 1
    assert "steps_per_sec" in r.stdout


def test_num_envs_change_flagged(tmp_path):
    r = _run(tmp_path, {**_HEALTHY_TRAIN, "num_envs": 512})
    assert r.returncode == 1
