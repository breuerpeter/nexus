"""The Benchmarking page, which the docs hook renders from the committed data and the examples baselines."""

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _hook():
    spec = importlib.util.spec_from_file_location("benchmarks", ROOT / "docs" / "hooks" / "benchmarks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_built_page_plots_a_metric_when_the_baselines_hold_a_rule_for_it():
    """The built page plots a metric when `scripts/ci/examples_baselines.json` holds a rule for it, and
    carries that rule's bound.

    Given a baselines file with a rule for `rtf` of `pid` and none for its `final_err_m`, when the docs
    build, then the built Benchmarking page names `rtf[pid]` with its bound and leaves out
    `final_err_m[pid]`. The rule is at least 0.8 of a recorded 5.0, so the bound is a `min` of 4.0. The
    page carries the bounds as one JSON block, which the page's script reads.
    """
    baselines = {"_meta": {"hardware": "a box"}, "pid": {"rtf": {"value": 5.0, "min_frac": 0.8}}}

    section = _hook().trends(baselines)

    block = re.search(r'<script type="application/json"[^>]*>(.*?)</script>', section, re.S)
    assert json.loads(block.group(1)) == {"rtf[pid]": {"min": 4.0}}


def test_the_examples_table_still_builds_from_the_committed_data_with_its_one_stamp():
    """The examples table on the Benchmarking page still builds from `docs/data/examples_bench.json`
    with its one `_meta` stamp.

    Given the committed docs data, when the docs build, then the Benchmarking page holds one row per
    example and the stamp with the sha, the date and the GPU.
    """
    data = json.loads((ROOT / "docs" / "data" / "examples_bench.json").read_text())
    examples = {entry["name"].split("[")[1].rstrip("]") for entry in data["entries"]}
    expected = [f"| `{example}` |" for example in examples] + list(data["_meta"].values())

    page = _hook().on_page_markdown("<!-- benchmark-examples -->\n", page=None, config=None, files=None)

    assert expected and [text for text in expected if text not in page] == []
