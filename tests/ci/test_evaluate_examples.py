"""The examples evaluation harness, scripts/ci/evaluate_examples.py: CPU CI exercises its gate evaluator,
waypoint reference interpolation, and the committed baselines, while the flights that produce fresh
artifacts run on the GPU runner. The GPU runner also covers evo-dependent scoring implicitly, since the
`ci` dependency-group isn't installed here.
"""

import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
import time

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASELINES = json.loads((ROOT / "scripts" / "ci" / "examples_baselines.json").read_text())
_REAL_RUN = subprocess.run  # the real one, kept before a test fakes the flights

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


def test_a_bench_feed_read_s3_denies_fails_the_upload_naming_the_key(tmp_path, monkeypatch):
    """A bench-feed read that S3 denies still reds the job, naming the key: given
    `evaluate_examples.py --upload` with an `aws` on `PATH` that answers `get-object` with
    `AccessDenied`, when it publishes the bench feed, then it exits non-zero and its error names the
    key and the denial.
    """
    aws = tmp_path / "bin" / "aws"
    aws.parent.mkdir()
    aws.write_text(
        "#!/bin/sh\n"
        'case "$*" in *get-object*)\n'
        "  echo 'An error occurred (AccessDenied) when calling the GetObject operation' >&2; exit 254;;\n"
        "esac\n"
    )
    aws.chmod(0o755)
    monkeypatch.setenv("PATH", f"{aws.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("NEXUS_BUCKET", "ci-bucket")
    monkeypatch.setenv("GITHUB_SHA", "0123456789abcdef0123456789abcdef01234567")
    real_run = subprocess.run
    _fake_flight(monkeypatch, _HEALTHY_FLIGHT)
    flight = subprocess.run
    # The flight stays faked; the harness's `aws` calls reach the fake `aws` on PATH.
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: (real_run if cmd[0] == "aws" else flight)(cmd, **kw))

    with pytest.raises(RuntimeError, match=r"get-object public/ci/bench/0123456789ab\.json: .*AccessDenied"):
        _harness(monkeypatch, "--only", "goto_policy", "--out", str(tmp_path / "out"), "--upload")


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


# A flight inside every committed guard of its example, rtf included, for the examples a CPU box can fly.
_HEALTHY = {
    "pid": {"reached": 4, "final_dist_m": 0.1, "ape_trans_rmse_m": 0.8, "rtf": 6.7},
    "gain_tuning": {"gradient_cosine": 0.995, "opt_hold_m": 0.1, "rtf": 6.3},
    "mass_recovery": {"mass_rel_err": 0.01, "wall_s": 34.0},
    "sampling_mpc": {
        "reached": 3,
        "final_dist_m": 0.1,
        "min_clearance_m": 0.5,
        "max_tilt_deg": 30.0,
        "path_wiggle": 1.0,
        "rtf": 0.29,
    },
    "goto_policy": {**_HEALTHY_FLIGHT, "rtf": 2.1},
}


def _fake_flights(monkeypatch, recordings: pathlib.Path, flights: dict[str, tuple[dict, int]]) -> None:
    """Stand in for each example process at the process boundary, as `_fake_flight` does, for each named
    example at once: each writes its dump and a recording, as an example does, then exits with its code.
    """

    def run(cmd, **kwargs):
        if cmd[:2] == ["uv", "run"]:
            example = cmd[cmd.index("nexus.examples") + 1]
            stats, rc = flights[example]
            rrd = recordings / f"{example}.rrd"
            rrd.parent.mkdir(parents=True, exist_ok=True)
            rrd.write_bytes(b"a recording")
            out = pathlib.Path(kwargs["env"]["NEXUS_EVAL_OUT"])
            dump = {"stats": stats, "results": {"rtf": stats.get("rtf", 2.0)}, "rrd": str(rrd)}
            (out / f"{example}.json").write_text(json.dumps(dump))
            return subprocess.CompletedProcess(cmd, rc, "", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)


def _scored(out: pathlib.Path) -> set[str]:
    """The examples `benchmark.json` holds scores for, read from its `metric[example]` entry names."""
    return {e["name"].split("[")[1].rstrip("]") for e in json.loads((out / "benchmark.json").read_text())}


def test_on_a_pull_request_an_examples_rtf_never_fails_the_leg(tmp_path, monkeypatch, capsys):
    """On a pull request an example's `rtf` never fails the leg, and the baselines file keeps its `rtf`
    rows: given the committed `examples_baselines.json` and a `pid` flight at half its recorded `rtf`,
    when the harness gates it as a pull-request run, then the table shows `rtf` as `(monitored)` and the
    run exits 0.
    """
    _fake_flights(monkeypatch, tmp_path / "rec", {"pid": ({**_HEALTHY["pid"], "rtf": 3.37}, 0)})  # half of 6.749

    rc = _harness(monkeypatch, "--only", "pid", "--shared", "--out", str(tmp_path / "out"))

    assert (rc, "rtf" in _rows(capsys.readouterr().out, "(monitored)")) == (0, True)


def test_on_a_pull_request_a_regressed_correctness_row_still_fails_the_leg(tmp_path, monkeypatch, capsys):
    """On a pull request a regressed correctness row still fails the leg: given the committed baselines
    and a `pid` flight that reaches fewer waypoints than recorded, when the harness gates it as a
    pull-request run, then `reached` shows `REGRESSED` and the run exits 1.
    """
    _fake_flights(monkeypatch, tmp_path / "rec", {"pid": ({**_HEALTHY["pid"], "reached": 3}, 0)})  # of 4

    rc = _harness(monkeypatch, "--only", "pid", "--shared", "--out", str(tmp_path / "out"))

    assert (rc, "reached" in _rows(capsys.readouterr().out, "REGRESSED")) == (1, True)


def test_on_a_pull_request_one_red_example_fails_the_leg_and_leaves_the_others_scores(tmp_path, monkeypatch, capsys):
    """On a pull request one red example fails the leg and leaves the other six's scores: given the
    pull-request shape with one example's flight faked to exit non-zero and six green, when it runs, then
    it exits non-zero, lists that example under `FAILED examples`, and the joined output holds the six
    others' scores. Here with the five examples a CPU box can fly, one red and four green.
    """
    flights = {name: (stats, 0) for name, stats in _HEALTHY.items()}
    flights["mass_recovery"] = (_HEALTHY["mass_recovery"], 1)
    _fake_flights(monkeypatch, tmp_path / "rec", flights)
    out = tmp_path / "out"

    rc = _harness(monkeypatch, "--only", ",".join(flights), "--shared", "--out", str(out))

    failed = "FAILED examples: mass_recovery" in capsys.readouterr().out
    assert (rc, failed, _scored(out)) == (1, True, {"pid", "gain_tuning", "sampling_mpc", "goto_policy"})


def test_on_a_pull_request_a_green_examples_recording_stays_out_of_the_artifact(tmp_path, monkeypatch):
    """On a pull request a green example's recording stays out of the artifact: given a green `pid`
    flight that wrote an `.rrd`, when the harness runs it as a pull-request run, then the output holds its
    scores and dump but no `pid.rrd`.
    """
    _fake_flights(monkeypatch, tmp_path / "rec", {"pid": (_HEALTHY["pid"], 0)})
    out = tmp_path / "out"

    _harness(monkeypatch, "--only", "pid", "--shared", "--out", str(out))

    kept = {p.name for p in out.iterdir()}
    assert ({"pid.json", "benchmark.json"} <= kept, "pid.rrd" in kept) == (True, False)


def test_on_a_pull_request_an_example_that_fails_a_gate_carries_its_recording(tmp_path, monkeypatch):
    """On a pull request an example that fails a gate carries its recording: given a `pid` flight that
    wrote an `.rrd` and regressed a correctness row, when the harness runs it as a pull-request run, then
    the output holds `pid.rrd`.
    """
    _fake_flights(monkeypatch, tmp_path / "rec", {"pid": ({**_HEALTHY["pid"], "reached": 3}, 0)})
    out = tmp_path / "out"

    _harness(monkeypatch, "--only", "pid", "--shared", "--out", str(out))

    assert (out / "pid.rrd").is_file()


def test_on_a_pull_request_an_example_whose_run_fails_carries_its_recording(tmp_path, monkeypatch):
    """On a pull request an example whose run fails carries its recording when it wrote one: given a `pid`
    flight that wrote an `.rrd` and exited non-zero, when the harness runs it as a pull-request run, then
    the output holds `pid.rrd`.
    """
    _fake_flights(monkeypatch, tmp_path / "rec", {"pid": (_HEALTHY["pid"], 1)})
    out = tmp_path / "out"

    _harness(monkeypatch, "--only", "pid", "--shared", "--out", str(out))

    assert (out / "pid.rrd").is_file()


def test_on_main_an_rtf_under_its_ratchet_still_fails_the_leg(tmp_path, monkeypatch, capsys):
    """On main an `rtf` under its ratchet still fails the leg: given the committed baselines and a `pid`
    flight at half its recorded `rtf`, when the harness gates it as a main run, then `rtf` shows
    `REGRESSED` and the run exits 1.
    """
    _fake_flights(monkeypatch, tmp_path / "rec", {"pid": ({**_HEALTHY["pid"], "rtf": 3.37}, 0)})  # half of 6.749

    rc = _harness(monkeypatch, "--only", "pid", "--out", str(tmp_path / "out"))

    assert (rc, "rtf" in _rows(capsys.readouterr().out, "REGRESSED")) == (1, True)


def test_on_main_a_green_flights_recording_stays_out_of_the_github_artifact(tmp_path, monkeypatch):
    """On main a green flight's recording stays out of the GitHub artifact: given a green `pid` flight
    that wrote an `.rrd`, when the harness runs it as a main run, then the output holds its scores and
    `benchmark.json` but no `pid.rrd`.
    """
    _fake_flights(monkeypatch, tmp_path / "rec", {"pid": (_HEALTHY["pid"], 0)})
    out = tmp_path / "out"

    _harness(monkeypatch, "--only", "pid", "--out", str(out))

    kept = {p.name for p in out.iterdir()}
    assert ({"pid.json", "benchmark.json"} <= kept, "pid.rrd" in kept) == (True, False)


def test_on_main_a_flight_that_fails_a_gate_carries_its_recording(tmp_path, monkeypatch):
    """On main a flight that fails a gate carries its recording: given a `pid` flight that wrote an
    `.rrd` and regressed a correctness row, when the harness runs it as a main run, then the output
    holds `pid.rrd`.
    """
    _fake_flights(monkeypatch, tmp_path / "rec", {"pid": ({**_HEALTHY["pid"], "reached": 3}, 0)})  # of 4
    out = tmp_path / "out"

    _harness(monkeypatch, "--only", "pid", "--out", str(out))

    assert (out / "pid.rrd").is_file()


def test_on_main_a_flight_whose_run_fails_carries_its_recording(tmp_path, monkeypatch):
    """On main a flight whose run fails carries its recording: given a `pid` flight that wrote an `.rrd`
    and exited non-zero, when the harness runs it as a main run, then the output holds `pid.rrd`.
    """
    _fake_flights(monkeypatch, tmp_path / "rec", {"pid": (_HEALTHY["pid"], 1)})
    out = tmp_path / "out"

    _harness(monkeypatch, "--only", "pid", "--out", str(out))

    assert (out / "pid.rrd").is_file()


def _stand_in_aws(monkeypatch, tmp_path: pathlib.Path) -> pathlib.Path:
    """Put a stand-in `aws` on `PATH` for an `--upload` run and return the file it logs to, one line of
    arguments per call. It finds no bench feed and accepts every write. Call it after faking the
    flights: they stay faked, and the harness's `aws` calls reach the stand-in.
    """
    log = tmp_path / "aws.log"
    aws = tmp_path / "bin" / "aws"
    aws.parent.mkdir()
    aws.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> {log}\n'
        'case "$*" in *get-object*)\n'
        "  echo 'An error occurred (NoSuchKey) when calling the GetObject operation' >&2; exit 254;;\n"
        "esac\n"
    )
    aws.chmod(0o755)
    monkeypatch.setenv("PATH", f"{aws.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("NEXUS_BUCKET", "ci-bucket")
    monkeypatch.setenv("GITHUB_SHA", "0123456789abcdef0123456789abcdef01234567")
    real_run, flight = _REAL_RUN, subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: (real_run if cmd[0] == "aws" else flight)(cmd, **kw))
    return log


def _recordings_copied(log: pathlib.Path) -> set[str]:
    """The S3 destinations of the `.rrd` files the logged `aws s3 cp` calls copied."""
    calls = [line.split() for line in log.read_text().splitlines()] if log.exists() else []
    return {call[3] for call in calls if call[:2] == ["s3", "cp"] and call[3].endswith(".rrd")}


def test_an_upload_run_writes_a_recording_to_s3_only_under_the_key_the_docs_serve(tmp_path, monkeypatch):
    """An upload run writes a recording to S3 only under the key the docs serve: given a green `pid`
    flight that wrote an `.rrd`, when the harness runs with `--upload` against a stand-in `aws`, then it
    copies the recording to `public/ci/logs/pid.rrd` and to no key under `public/ci/logs/<sha12>/`.
    """
    _fake_flights(monkeypatch, tmp_path / "rec", {"pid": (_HEALTHY["pid"], 0)})
    log = _stand_in_aws(monkeypatch, tmp_path)

    _harness(monkeypatch, "--only", "pid", "--out", str(tmp_path / "out"), "--upload")

    assert _recordings_copied(log) == {"s3://ci-bucket/public/ci/logs/pid.rrd"}


def test_an_upload_run_writes_no_recording_the_docs_do_not_serve(tmp_path, monkeypatch):
    """An upload run writes no recording the docs don't serve: given a green `goto_policy_fresh` flight
    that wrote an `.rrd`, when the harness runs with `--upload` against a stand-in `aws`, then it copies
    no `.rrd` to S3.
    """
    policy = tmp_path / "policy.pt"
    policy.write_bytes(b"a fresh export")
    # The fresh flight runs the `goto_policy` launcher, which is the name its process carries.
    _fake_flights(monkeypatch, tmp_path / "rec", {"goto_policy": (_HEALTHY["goto_policy"], 0)})
    log = _stand_in_aws(monkeypatch, tmp_path)

    _harness(
        monkeypatch, "--only", "goto_policy_fresh", "--policy", str(policy), "--out", str(tmp_path / "out"), "--upload"
    )

    assert _recordings_copied(log) == set()


def test_an_upload_run_still_writes_the_bench_feed_under_its_per_commit_key_and_latest(tmp_path, monkeypatch):
    """An upload run still writes the bench feed under its per-commit key and its fixed key: given a
    green `pid` flight with scores, when the harness runs with `--upload` against a stand-in `aws`, then
    it writes `public/ci/bench/<sha12>.json` and the fixed key beside it.
    """
    _fake_flights(monkeypatch, tmp_path / "rec", {"pid": (_HEALTHY["pid"], 0)})
    log = _stand_in_aws(monkeypatch, tmp_path)

    _harness(monkeypatch, "--only", "pid", "--out", str(tmp_path / "out"), "--upload")

    puts = [line.split() for line in log.read_text().splitlines() if "put-object" in line]
    written = {call[call.index("--key") + 1] for call in puts}
    assert written == {"public/ci/bench/0123456789ab.json", "public/ci/bench/latest.json"}


_INDEX = "public/ci/bench/latest.json"
_THIS_COMMIT = "public/ci/bench/0123456789ab.json"  # the key of the commit `_stand_in_s3` uploads at
_OTHER_COMMIT = "public/ci/bench/aaaaaaaaaaaa.json"
_A_DATE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")

# A stand-in `aws` that holds its objects in a folder, one file per key. A key with a `.raced` file
# beside it refuses its next write as S3 does when another box wrote since the read, and that file
# becomes the object.
_S3 = """#!PYTHON
import hashlib, json, pathlib, shutil, sys

store, args = pathlib.Path("STORE"), sys.argv[1:]
if args[:2] == ["s3api", "get-object"]:
    held = store / args[args.index("--key") + 1]
    if not held.is_file():
        sys.exit("An error occurred (NoSuchKey) when calling the GetObject operation")
    shutil.copy(held, args[-1])
    print(json.dumps({"ETag": hashlib.md5(held.read_bytes()).hexdigest()}))
elif args[:2] == ["s3api", "put-object"]:
    held = store / args[args.index("--key") + 1]
    raced = held.with_name(held.name + ".raced")
    if raced.is_file():
        raced.replace(held)
        sys.exit("An error occurred (PreconditionFailed) when calling the PutObject operation")
    held.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(args[args.index("--body") + 1], held)
"""


def _stand_in_s3(monkeypatch, tmp_path: pathlib.Path, held: dict, raced: dict | None = None) -> pathlib.Path:
    """Put a stand-in `aws` on `PATH` for an `--upload` run at commit `0123456789ab` and return the
    folder it holds its objects in. It starts with the objects of *held*, each key with its JSON
    content, and reads and writes them as S3 would. Each key of *raced* refuses its next write with
    `PreconditionFailed` and then holds that content, as when another box wrote first. Call it after
    faking the flights: they stay faked, and the harness's `aws` calls reach the stand-in.
    """
    store = tmp_path / "s3"
    for key, content in held.items():
        (store / key).parent.mkdir(parents=True, exist_ok=True)
        (store / key).write_text(json.dumps(content))
    for key, content in (raced or {}).items():
        (store / key).parent.mkdir(parents=True, exist_ok=True)
        (store / f"{key}.raced").write_text(json.dumps(content))
    aws = tmp_path / "bin" / "aws"
    aws.parent.mkdir()
    aws.write_text(_S3.replace("PYTHON", sys.executable).replace("STORE", str(store)))
    aws.chmod(0o755)
    monkeypatch.setenv("PATH", f"{aws.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("NEXUS_BUCKET", "ci-bucket")
    monkeypatch.setenv("GITHUB_SHA", "0123456789abcdef0123456789abcdef01234567")
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)  # the wait before a refused write's retry
    real_run, flight = _REAL_RUN, subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: (real_run if cmd[0] == "aws" else flight)(cmd, **kw))
    return store


def _upload_a_green_pid_flight(monkeypatch, tmp_path: pathlib.Path, held: dict, raced: dict | None = None):
    """Run the harness with `--upload` on one green `pid` flight against a stand-in S3 that holds *held*,
    and return the harness's exit code and the content of the fixed key, `_INDEX`, after it.
    """
    _fake_flights(monkeypatch, tmp_path / "rec", {"pid": (_HEALTHY["pid"], 0)})
    store = _stand_in_s3(monkeypatch, tmp_path, held, raced)
    code = _harness(monkeypatch, "--only", "pid", "--out", str(tmp_path / "out"), "--upload")
    return code, json.loads((store / _INDEX).read_text())


def _listed(index) -> list:
    """The per-commit keys *index* lists, in its order, with `None` for an entry that names no key."""
    return [entry.get("key") if isinstance(entry, dict) else None for entry in index]


def test_an_upload_run_adds_its_commit_to_the_index_with_its_key_and_its_date(tmp_path, monkeypatch):
    """An `--upload` run adds its commit to the index at the fixed key: the key of its
    `bench/<sha12>.json` and its date. Given a stand-in S3 whose index lists one commit, when a harness
    run with `--upload` ends at a new commit, then the index lists both commits, each with its key and
    its date.
    """
    held = {_INDEX: [{"key": _OTHER_COMMIT, "date": "2026-10-01T08:00:00Z"}]}

    _, index = _upload_a_green_pid_flight(monkeypatch, tmp_path, held)

    dated = {e.get("key"): bool(_A_DATE.fullmatch(str(e.get("date")))) for e in index if isinstance(e, dict)}
    assert dated == {_OTHER_COMMIT: True, _THIS_COMMIT: True}


def test_the_first_upload_run_against_a_bucket_with_no_index_creates_it(tmp_path, monkeypatch):
    """The first `--upload` run against a bucket with no index creates it: given a stand-in S3 with no
    object at the fixed key, when a harness run with `--upload` ends, then the index exists and lists that
    one commit.
    """
    _, index = _upload_a_green_pid_flight(monkeypatch, tmp_path, held={})

    assert _listed(index) == [_THIS_COMMIT]


def test_an_upload_run_that_finds_the_old_snapshot_replaces_it_with_the_index(tmp_path, monkeypatch):
    """An `--upload` run that finds the old snapshot at the fixed key, a bare list of entries,
    replaces it with the index: given a stand-in S3 whose fixed key holds a list of entries, when
    a harness run with `--upload` ends, then the key holds an index that lists the commit, and the run
    exits as it would have.
    """
    snapshot = [{"name": "rtf[px4_sitl]", "unit": "x realtime", "value": 2.2, "biggerIsBetter": True}]

    code, index = _upload_a_green_pid_flight(monkeypatch, tmp_path, held={_INDEX: snapshot})

    assert (code, _listed(index)) == (0, [_THIS_COMMIT])


def test_a_second_upload_at_a_commit_the_index_lists_leaves_it_listed_once(tmp_path, monkeypatch):
    """A second upload at a commit the index already lists leaves that commit listed once: given a
    stand-in S3 whose index lists the commit, when a harness run with `--upload` ends at that commit,
    then the index lists it once.
    """
    held = {_INDEX: [{"key": _THIS_COMMIT, "date": "2026-10-01T08:00:00Z"}]}

    _, index = _upload_a_green_pid_flight(monkeypatch, tmp_path, held)

    assert _listed(index) == [_THIS_COMMIT]


def test_two_boxes_that_upload_at_once_both_land_in_the_index(tmp_path, monkeypatch):
    """Two boxes that upload at once both land in the index: given a stand-in S3 that refuses the first
    conditional write with `PreconditionFailed` after another box wrote a commit, when the upload runs,
    then the index lists both commits.
    """
    raced = {_INDEX: [{"key": _OTHER_COMMIT, "date": "2026-10-01T08:00:00Z"}]}

    _, index = _upload_a_green_pid_flight(monkeypatch, tmp_path, held={}, raced=raced)

    assert sorted(_listed(index), key=str) == [_THIS_COMMIT, _OTHER_COMMIT]
