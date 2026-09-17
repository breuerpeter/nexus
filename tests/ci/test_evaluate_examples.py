"""The examples evaluation harness, scripts/ci/evaluate_examples.py: CPU CI exercises its gate evaluator,
waypoint reference interpolation, and the committed baselines, while the flights that produce fresh
artifacts run on the GPU runner. The GPU runner also covers evo-dependent scoring implicitly, since the
`ci` dependency-group isn't installed here.
"""

import importlib.util
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASELINES = json.loads((ROOT / "scripts" / "ci" / "examples_baselines.json").read_text())

_spec = importlib.util.spec_from_file_location("evaluate_examples", ROOT / "scripts" / "ci" / "evaluate_examples.py")
evaluate_examples = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_spec and evaluate_examples)


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
