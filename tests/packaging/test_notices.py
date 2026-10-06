"""A file that names another owner or license points at an entry in the notices.

The project is Apache-2.0 by Peter Breuer. A tracked file with a ``Copyright`` or ``SPDX-License-Identifier``
line that names another copyright holder, or a license other than Apache-2.0, came from another project, and
``THIRD_PARTY_NOTICES.txt`` names it. The test reads every line, not a header of fixed length, since a header
can run long and a bundled file repeats its notices through its body. It reads the tracked tree, so it needs
no GPU and no container.
"""

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NOTICES = REPO / "THIRD_PARTY_NOTICES.txt"

OWNER = "Peter Breuer"
LICENSE = "Apache-2.0"
# The license texts themselves name owners and licenses by design.
LICENSE_FILES = {"LICENSE", "THIRD_PARTY_NOTICES.txt"}

COPYRIGHT = re.compile(r"Copyright\s+(?:\(c\)\s*|©\s*)?\d{4}")
SPDX = re.compile(r"SPDX-License-Identifier:\s*([\w.+-]+)")


def _tracked_text_files() -> list[str]:
    """Every tracked file that holds text, as a repo-relative path."""
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    files = []
    for f in out.splitlines():
        path = REPO / f
        if f in LICENSE_FILES or not path.is_file():
            continue
        head = path.read_bytes()[:8192]
        if b"\0" not in head:
            files.append(f)
    return files


def _claims_another_owner_or_license(path: Path) -> bool:
    """Whether a line of the file names a copyright holder other than the project's, or another license."""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if COPYRIGHT.search(line) and OWNER not in line:
            return True
        spdx = SPDX.search(line)
        if spdx and spdx.group(1) != LICENSE:
            return True
    return False


def _noticed_files() -> set[str]:
    """The files ``THIRD_PARTY_NOTICES.txt`` names: each entry is a path underlined with dashes."""
    lines = NOTICES.read_text().splitlines()
    return {lines[i].strip() for i in range(len(lines) - 1) if lines[i].strip() and set(lines[i + 1].strip()) == {"-"}}


def test_a_header_naming_another_owner_or_license_is_named_in_the_notices() -> None:
    """No tracked file's header claims an owner or license other than this project's unless
    `THIRD_PARTY_NOTICES.txt` names that file.
    """
    noticed = _noticed_files()
    foreign = [f for f in _tracked_text_files() if _claims_another_owner_or_license(REPO / f)]
    assert [f for f in foreign if f not in noticed] == []
