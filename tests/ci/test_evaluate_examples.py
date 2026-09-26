"""The examples evaluation harness, scripts/ci/evaluate_examples.py: CPU CI exercises its gate evaluator,
waypoint reference interpolation, and the committed baselines, while the flights that produce fresh
artifacts run on the GPU runner. The GPU runner also covers evo-dependent scoring implicitly, since the
`ci` dependency-group isn't installed here.
"""

import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASELINES = json.loads((ROOT / "scripts" / "ci" / "examples_baselines.json").read_text())

# Loaded as it runs, `python scripts/ci/evaluate_examples.py`: its own folder first on sys.path,
# where its sibling merge_eval_parts lives.
sys.path.insert(0, str(ROOT / "scripts" / "ci"))
_spec = importlib.util.spec_from_file_location("evaluate_examples", ROOT / "scripts" / "ci" / "evaluate_examples.py")
evaluate_examples = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_spec and evaluate_examples)

# A tour the hosted policy flies: every waypoint reached, on the path the runner recorded.
_HEALTHY_FLIGHT = {
    "waypoints": 3,
    "reached": 3,
    "control_steps": 1560,
    "deploy_steps_per_sec": 150.0,
    "final_tracking_error_m": 0.02,
    "ape_trans_rmse_m": BASELINES["goto_policy"]["ape_trans_rmse_m"]["value"],
    "ape_trans_max_m": 0.9,
}


def _fake_flight(monkeypatch, stats: dict, rtf: float = 2.0) -> None:
    """Stand in for the example process at the process boundary: the flight's dump lands where the
    harness points the example, named after the example as it would write it, and every other
    process the harness asks for answers empty.
    """

    def run(cmd, **kwargs):
        if cmd[:2] == ["uv", "run"]:
            example = cmd[cmd.index("nexus.examples") + 1]
            out = pathlib.Path(kwargs["env"]["NEXUS_EVAL_OUT"])
            (out / f"{example}.json").write_text(json.dumps({"stats": stats, "results": {"rtf": rtf}}))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)


def _harness(monkeypatch, *argv: str) -> int:
    """Run the harness's command line in-process and return its exit code."""
    monkeypatch.setattr(sys, "argv", ["evaluate_examples.py", *argv])
    return evaluate_examples.main()


def _rows(out: str, status: str) -> set[str]:
    """The metric names of the gate-table rows shown with *status*."""
    return {line.split()[0] for line in out.splitlines() if status in line.split()[-2:]}


def test_a_deploy_flights_wall_clock_numbers_never_fail_a_leg(capsys):
    """A deploy flight's wall-clock numbers never fail a leg: given the committed
    `scripts/ci/examples_baselines.json` and a `goto_policy` flight scored at half the recorded
    `deploy_steps_per_sec` and `rtf`, when the harness gates it, then the table shows both as
    `(monitored)` and the run passes.
    """
    slow = {**_HEALTHY_FLIGHT, "deploy_steps_per_sec": 74.9, "rtf": 1.07}  # half of the recorded 149.8 and 2.135

    failed = evaluate_examples._gate("goto_policy", slow, BASELINES)

    monitored = _rows(capsys.readouterr().out, "(monitored)")
    assert (failed, {"deploy_steps_per_sec", "rtf"} <= monitored) == ([], True)


def test_the_wall_clock_numbers_still_reach_the_trend_record(tmp_path, monkeypatch):
    """The wall-clock numbers still reach the trend record: given a scored deploy flight, when the
    harness writes `benchmark.json` and `examples_bench.json`, then they carry its
    `deploy_steps_per_sec` and `rtf` entries as today.
    """
    _fake_flight(monkeypatch, _HEALTHY_FLIGHT)
    out = tmp_path / "out"

    _harness(monkeypatch, "--only", "goto_policy", "--out", str(out))

    trend = {e["name"] for e in json.loads((out / "benchmark.json").read_text())}
    docs = {e["name"] for e in json.loads((out / "examples_bench.json").read_text())["entries"]}
    assert {"deploy_steps_per_sec[goto_policy]", "rtf[goto_policy]"} <= trend & docs


def test_the_fresh_flight_gates_on_completing_only(tmp_path, monkeypatch, capsys):
    """The fresh policy's flight gates on completing only: given the fresh policy's flight scored at 1 of
    3 reached, `final_tracking_error_m` 3.1 and an `ape_trans_rmse_m` over the hosted flight's bound,
    when the harness gates it, then every metric shows `(monitored)` and the run passes.
    """
    policy = tmp_path / "policy.pt"
    policy.write_bytes(b"a fresh export")
    one_of_three = {**_HEALTHY_FLIGHT, "reached": 1, "final_tracking_error_m": 3.1, "ape_trans_rmse_m": 5.0}
    _fake_flight(monkeypatch, one_of_three)

    rc = _harness(monkeypatch, "--only", "goto_policy_fresh", "--policy", str(policy), "--out", str(tmp_path / "out"))

    out = capsys.readouterr().out
    gated = _rows(out, "ok") | _rows(out, "REGRESSED")
    assert (rc, gated, {"reached", "final_tracking_error_m", "ape_trans_rmse_m"} <= _rows(out, "(monitored)")) == (
        0, set(), True,
    )  # fmt: skip


def test_a_fresh_export_the_deploy_side_cannot_fly_fails_the_leg(tmp_path, monkeypatch, capsys, torchscript_policy):
    """A fresh export the deploy side can't fly fails the leg: given an export whose observation width
    isn't the deploy controller's, a 12-input policy, when the harness runs the fresh policy's flight,
    then it lists the flight under `FAILED examples` and exits 1.
    """
    narrow = torchscript_policy(obs_dim=12)

    rc = _harness(monkeypatch, "--only", "goto_policy_fresh", "--policy", narrow, "--out", str(tmp_path / "out"))

    assert (rc, "FAILED examples: goto_policy_fresh" in capsys.readouterr().out) == (1, True)


def test_a_hosted_policy_the_runner_cannot_fetch_fails_the_leg(tmp_path, monkeypatch, capfd):
    """A hosted policy the runner can't fetch fails the leg: given the catalog's base unreachable, a
    dead proxy in the environment and an empty asset cache, when the harness runs the hosted policy's
    flight, then it lists the flight under `FAILED examples`, the log carries the `--policy` hint, and
    it exits 1.
    """
    monkeypatch.setenv("NEXUS_ASSET_CACHE", str(tmp_path / "cache"))  # empty: nothing to hit
    for proxy in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
        monkeypatch.setenv(proxy, "http://127.0.0.1:9")  # the discard port: no proxy answers

    rc = _harness(monkeypatch, "--only", "goto_policy", "--out", str(tmp_path / "out"))

    out, err = capfd.readouterr()
    hinted = "--policy" in err and "train.py" in err
    assert (rc, "FAILED examples: goto_policy" in out, hinted) == (1, True, True)


def test_baselines_cover_every_example():
    assert set(evaluate_examples.EXAMPLES) <= set(BASELINES) - {"_meta"}
    assert set(evaluate_examples.DEFAULT_SET) <= set(evaluate_examples.EXAMPLES)


def test_check_exact_match():
    ok, _ = evaluate_examples._check(3, 3)
    assert ok
    ok, _ = evaluate_examples._check(True, False)
    assert not ok


def test_check_absolute_guards():
    ok, _ = evaluate_examples._check({"max": 0.1}, 0.05)
    assert ok
    ok, _ = evaluate_examples._check({"max": 0.1}, 0.2)
    assert not ok
    ok, _ = evaluate_examples._check({"min": 1.0}, 0.5)
    assert not ok


def test_check_relative_guards_skip_while_unbaselined():
    rule = {"value": None, "min_frac": 0.8}
    ok, desc = evaluate_examples._check(rule, 0.0001)  # would fail any real floor
    assert ok and "unbaselined" in desc
    ok, _ = evaluate_examples._check({"value": 10.0, "min_frac": 0.8}, 7.0)
    assert not ok
    ok, _ = evaluate_examples._check({"value": 10.0, "min_frac": 0.8}, 9.0)
    assert ok


def test_check_combines_absolute_and_relative():
    rule = {"value": 4.7, "min": 2.0, "min_frac": 0.8}
    ok, _ = evaluate_examples._check(rule, 4.0)
    assert ok
    ok, _ = evaluate_examples._check(rule, 3.0)  # over the absolute floor, under 80% of value
    assert not ok


def test_waypoint_reference_interpolates_and_holds():
    est_t = np.linspace(0.0, 10.0, 101)
    est_pos = np.zeros((101, 3))
    wps = np.array([[2.0, 0.0, 1.0], [2.0, 4.0, 1.0]])
    arrivals = np.array([4.0, 8.0])
    ref = evaluate_examples._waypoint_reference(est_t, est_pos, wps, arrivals)
    np.testing.assert_allclose(ref[0], [0, 0, 0], atol=1e-12)  # anchored at the start pose
    np.testing.assert_allclose(ref[40], wps[0], atol=1e-9)  # at arrival_0 the reference IS wp0
    np.testing.assert_allclose(ref[20], [1.0, 0.0, 0.5], atol=1e-9)  # linear in time between knots
    np.testing.assert_allclose(ref[-1], wps[1], atol=1e-9)  # held at the last waypoint after arrival


def test_gate_flags_missing_metric():
    failed = evaluate_examples._gate(
        "goto_policy", {"reached": 3}, {"goto_policy": {"reached": 3, "rtf": {"min": 1.0}}}
    )
    assert failed == ["goto_policy.rtf (missing)"]


def test_the_gpu_name_reads_unknown_without_nvidia_smi(tmp_path, monkeypatch):
    """The bench feed's hardware note reads `unknown` on a box without nvidia-smi, such as CPU CI, and
    the harness carries on.
    """
    monkeypatch.setenv("PATH", str(tmp_path))  # an empty directory: no nvidia-smi anywhere on it

    assert evaluate_examples._gpu_name() == "unknown"


def test_merged_local_merges_by_name_fresh_winning(tmp_path):
    first = [{"name": "rtf[pid]", "value": 1.0}, {"name": "ape[pid]", "value": 0.1}]
    fresh = [{"name": "rtf[px4_sitl]", "value": 2.0}, {"name": "ape[pid]", "value": 0.2}]
    # The {_meta, entries} shape of examples_bench.json and the bare list of benchmark.json.
    wrapped = tmp_path / "examples_bench.json"
    wrapped.write_text(json.dumps({"_meta": {"sha": "old"}, "entries": first}))
    bare = tmp_path / "benchmark.json"
    bare.write_text(json.dumps(first))
    for path in (wrapped, bare):
        merged = {e["name"]: e["value"] for e in evaluate_examples._merged_local(path, fresh)}
        assert merged == {"rtf[pid]": 1.0, "rtf[px4_sitl]": 2.0, "ape[pid]": 0.2}
    # A missing or unreadable file degrades to the fresh entries alone.
    fresh_only = evaluate_examples._merged_local(tmp_path / "absent.json", fresh)
    assert sorted(e["name"] for e in fresh_only) == ["ape[pid]", "rtf[px4_sitl]"]
