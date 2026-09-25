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


def _recorded(key: str, fallback: float) -> float:
    """The baseline's recorded value for *key*, or *fallback* while the baseline has no such rule."""
    rule = BASELINE["train"].get(key)
    return rule["value"] if isinstance(rule, dict) and rule.get("value") is not None else fallback


# Healthy stats: exactly the baseline values, so this passes by construction whatever the pinned
# hardware numbers are. The baseline is re-recorded per CI runner, so no literals here. The learning
# metrics fall back to the values four seed-42 runs measured, until the runner records its own.
_HEALTHY_TRAIN = {
    "num_envs": BASELINE["train"]["num_envs"],
    "steps_per_sec": _recorded("steps_per_sec", 119515.0),
    "train_seconds": _recorded("train_seconds", 123.38),
    "success_rate": _recorded("success_rate", 0.99),
    "mean_reward": _recorded("mean_reward", 125.0),
}


def _rows(stdout: str, status: str) -> set[str]:
    """The metric names of the table rows that end in *status*."""
    return {line.split()[0] for line in stdout.splitlines() if line.strip().endswith(status)}


def test_baseline_is_well_formed():
    assert BASELINE["train"]["num_envs"] == 2048
    assert {"steps_per_sec", "train_seconds"}.isdisjoint(BASELINE["train"])  # wall clock prints, never gates
    assert "deploy" not in BASELINE  # deploy gates live in scripts/ci/examples_baselines.json now


def test_a_healthy_training_run_passes_the_leg(tmp_path):
    """A healthy training run passes the leg: given the committed `nexus-rl/stats_baseline.json` and a
    training stats file at the runner's recorded `success_rate` and mean reward, when
    `scripts/ci/check_rl_stats.py --train-stats` runs, then every row shows `ok` and it exits 0.
    """
    r = _run(tmp_path, _HEALTHY_TRAIN)

    shown = _rows(r.stdout, "ok")
    assert (r.returncode, _rows(r.stdout, "REGRESSED"), {"train.success_rate", "train.mean_reward"} <= shown) == (
        0, set(), True,
    ), r.stdout + r.stderr  # fmt: skip


def test_a_training_runs_wall_clock_numbers_never_fail_the_leg(tmp_path):
    """A training run's wall-clock numbers never fail the leg: given the committed
    `nexus-rl/stats_baseline.json` and a training stats file at half the recorded `steps_per_sec` and
    twice the `train_seconds`, when `scripts/ci/check_rl_stats.py --train-stats` runs, then it exits 0
    and prints both numbers.
    """
    slow = {
        **_HEALTHY_TRAIN,
        "steps_per_sec": _HEALTHY_TRAIN["steps_per_sec"] / 2,
        "train_seconds": _HEALTHY_TRAIN["train_seconds"] * 2,
    }

    r = _run(tmp_path, slow)

    assert (r.returncode, "steps_per_sec" in r.stdout, "train_seconds" in r.stdout) == (0, True, True), (
        r.stdout + r.stderr
    )


def test_a_training_run_that_fails_to_learn_fails_the_leg(tmp_path):
    """A training run that fails to learn fails the leg: given the committed `nexus-rl/stats_baseline.json`
    and a training stats file with `success_rate` 0.6 and a mean reward under the bound, when
    `scripts/ci/check_rl_stats.py --train-stats` runs, then the table shows `REGRESSED` on both and it
    exits 1.
    """
    unlearned = {**_HEALTHY_TRAIN, "success_rate": 0.6, "mean_reward": 60.0}

    r = _run(tmp_path, unlearned)

    assert (r.returncode, _rows(r.stdout, "REGRESSED")) == (1, {"train.success_rate", "train.mean_reward"}), (
        r.stdout + r.stderr
    )


def test_num_envs_change_flagged(tmp_path):
    r = _run(tmp_path, {**_HEALTHY_TRAIN, "num_envs": 512})
    assert r.returncode == 1
