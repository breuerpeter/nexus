"""What a release pull request leaves in the tree: release-please writes the next version to the files
its config lists, and to nothing else.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# The files `uv lock` reads for the two projects. Neither holds `.acados/`, as on a CI runner.
_PROJECT_FILES = ("pyproject.toml", "uv.lock", "nexus-rl/pyproject.toml", "nexus-rl/uv.lock")

# The two JSONPath shapes release-please documents for a project file and a uv lock, each as the text it selects: a
# dotted path to a table's key, and the `version` of the `[[package]]` entry with a given name.
_TABLE_KEY = re.compile(r"^\$\.(\w+)\.(\w+)$")
_NAMED_PACKAGE = re.compile(r"""^\$\.package\[\?\(@\.name\.value==["']([\w-]+)["']\)\]\.version$""")


def _write_version(file: Path, jsonpath: str, version: str) -> None:
    """Set the one value `jsonpath` selects in `file` to `version`, as release-please does."""
    if m := _TABLE_KEY.match(jsonpath):
        table, key = m.groups()
        selected = rf'(^\[{table}\]\n(?:(?!\[).*\n)*?{key} = ")[^"]+'
    elif m := _NAMED_PACKAGE.match(jsonpath):
        selected = rf'(^\[\[package\]\]\nname = "{m.group(1)}"\nversion = ")[^"]+'
    else:
        pytest.fail(f"the test reads no JSONPath of this shape: {jsonpath}")
    text, count = re.subn(selected, rf"\g<1>{version}", file.read_text(), flags=re.MULTILINE)
    assert count == 1, f"{jsonpath} selects {count} values in {file.name}"
    file.write_text(text)


@pytest.fixture
def released_tree(tmp_path):
    """A copy of the two projects with version 99.0.0 written to every file the release config lists."""
    for name in _PROJECT_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, tmp_path / name)
    config = json.loads((ROOT / "release-please-config.json").read_text())
    for extra in config["packages"]["."]["extra-files"]:
        _write_version(tmp_path / extra["path"], extra["jsonpath"], "99.0.0")
    return tmp_path


def _lock_check(tree: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["uv", "lock", "--check", *args], check=False, cwd=tree, capture_output=True, text=True)


def test_the_root_lock_is_fresh_after_a_release_bump(released_tree):
    """The root lock stays fresh under a release bump, on a checkout with no `.acados/`."""
    out = _lock_check(released_tree)
    assert out.returncode == 0, out.stderr


def test_the_rl_lock_is_fresh_after_a_release_bump(released_tree):
    """The `nexus-rl` lock stays fresh under the same bump."""
    out = _lock_check(released_tree, "--project", "nexus-rl")
    assert out.returncode == 0, out.stderr
