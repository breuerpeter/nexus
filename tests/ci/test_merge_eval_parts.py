"""The join of a fanned-out gpu-examples run's parts, scripts/ci/merge_eval_parts.py: what the merged
`examples-eval` artifact holds, given the parts each GPU box uploaded.
"""

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _part(parts: pathlib.Path, example: str, value: float) -> pathlib.Path:
    """Write the downloaded part of *example* as its box uploads it: its scores and a `benchmark.json`."""
    part = parts / f"examples-eval-{example}"
    part.mkdir(parents=True)
    (part / f"{example}.json").write_text(json.dumps({"stats": {"reached": 4}}))
    (part / "benchmark.json").write_text(json.dumps([{"name": f"rtf[{example}]", "value": value}]))
    return part


def _merge(parts: pathlib.Path, out: pathlib.Path) -> None:
    """Run the merge's command line as the merge job does."""
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ci" / "merge_eval_parts.py"), str(parts), str(out)], check=True
    )


def test_the_merged_artifact_joins_the_score_files_of_every_part(tmp_path):
    """The merged artifact joins the score files of every part: given two downloaded parts, each with its
    example's scores and a `benchmark.json`, when the merge runs, then the output holds both examples'
    scores and one `benchmark.json` with both entries.
    """
    _part(tmp_path / "parts", "pid", 6.7)
    _part(tmp_path / "parts", "gain_tuning", 6.3)
    out = tmp_path / "out"

    _merge(tmp_path / "parts", out)

    entries = {e["name"] for e in json.loads((out / "benchmark.json").read_text())}
    scores = {p.name for p in out.iterdir()} & {"pid.json", "gain_tuning.json"}
    assert (scores, entries) == ({"pid.json", "gain_tuning.json"}, {"rtf[pid]", "rtf[gain_tuning]"})


def test_a_red_flights_recording_rides_its_part_alone_not_the_merged_artifact_too(tmp_path):
    """A red flight's recording rides its part alone, not the merged artifact too: given a downloaded
    part that holds a red flight's `.rrd`, when the merge runs, then the output holds no `.rrd`.
    """
    (_part(tmp_path / "parts", "pid", 6.7) / "pid.rrd").write_bytes(b"a recording")
    out = tmp_path / "out"

    _merge(tmp_path / "parts", out)

    assert [p.name for p in out.rglob("*.rrd")] == []
