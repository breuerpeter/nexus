#!/usr/bin/env python3
"""Vale on the lines a branch adds: the check CI's vale-changed makes with reviewdog, run before the push.

Every alert, a suggestion included, on a line the branch adds to a Markdown or Python file fails the check, and
an alert on a line the branch never touched does not, so the alerts main still carries stay out of the way. The
lines are the working tree's against the merge base with ``origin/main``, or with ``--base``. The pre-commit hook
``vale-added`` runs it, so ``uvx pre-commit run -a`` runs it too.

    python3 scripts/ci/vale_added.py
    python3 scripts/ci/vale_added.py --base origin/main
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

# The files Vale reads, as .vale.ini scopes them, but the changelog release-please writes, as CI's glob leaves it out.
SUFFIXES = {".md", ".py"}
EXCLUDED = {"CHANGELOG.md"}
HUNK = re.compile(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def added_lines(base: str) -> dict[str, set[int]]:
    """The lines the working tree adds to each Markdown or Python file against the merge base with ``base``, by
    path: a new file's every line, and a changed file's added ones.
    """
    merge_base = _git("merge-base", base, "HEAD").strip()
    diff = _git("diff", "-U0", "--no-color", "--diff-filter=AMR", merge_base, "--", *(f"*{s}" for s in SUFFIXES))
    lines: dict[str, set[int]] = {}
    path = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            if Path(path).name in EXCLUDED:
                path = None
        elif path is not None and (hunk := HUNK.match(line)):
            start, count = int(hunk.group(1)), int(hunk.group(2) or 1)
            lines.setdefault(path, set()).update(range(start, start + count))
    return {p: ls for p, ls in lines.items() if ls}


def alerts_on(lines: dict[str, set[int]]) -> list[str]:
    """Vale's alerts on the ``lines`` of each file, one line each: the path, the line and column, the severity, the
    message and the rule.
    """
    if not lines:
        return []
    run = subprocess.run(["vale", "--output=JSON", *lines], check=False, capture_output=True, text=True)
    if run.returncode not in (0, 1) or not run.stdout.strip():  # 1 is Vale's exit on an error-level alert
        sys.exit(f"vale failed: {run.stderr.strip() or run.stdout.strip()}")
    out = []
    for path, found in json.loads(run.stdout).items():
        path = str(Path(path).resolve().relative_to(Path.cwd()))
        for alert in found:
            if alert["Line"] in lines.get(path, ()):
                out.append((path, alert["Line"], alert["Span"][0], alert["Severity"], alert["Message"], alert["Check"]))
    return [
        f"{path}:{line}:{col}  {severity}  {message}  ({check})"
        for path, line, col, severity, message, check in sorted(out)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="origin/main", help="the branch the diff is against, origin/main by default")
    args = parser.parse_args()
    if shutil.which("vale") is None:
        sys.exit("vale is not installed: see docs/guide/contributing/development.md, then run `vale sync` once")
    found = alerts_on(added_lines(args.base))
    for line in found:
        print(line)
    if found:
        print(f"{len(found)} vale alert(s) on lines the branch adds: CI's vale-changed fails on each", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
