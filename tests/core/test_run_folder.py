"""The folder that holds one run's files: a new one under the home folder's cache, or the one a caller names."""

import re

import pytest

from nexus_uav._src.core.run_folder import run_folder


def test_a_named_folder_that_holds_a_file_is_refused_by_name(tmp_path):
    """A named folder that holds a file is refused with an error that names it, and the file stays."""
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "old.txt").write_text("an earlier run's\n")

    with pytest.raises(FileExistsError, match=re.escape(str(folder))):
        run_folder(folder)
