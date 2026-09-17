"""Shared test helpers."""

import re
import subprocess
import sys

import pytest


@pytest.fixture(scope="session")
def rrd_entities():
    """Read the entity paths a written ``.rrd`` carries, through the bundled command-line tool.

    rerun 0.34 dropped the local dataframe API, ``rerun.recording.load_recording``; reading an rrd
    now needs the catalog server + the optional datafusion dep. The version-matched command-line
    tool still prints per-chunk entity paths, so tests parse those.
    """

    def _entities(path: str) -> list[str]:
        out = subprocess.run(
            [sys.executable, "-m", "rerun", "rrd", "print", path], capture_output=True, text=True, check=True
        ).stdout
        return sorted(set(re.findall(r" - (/\S*) - data columns:", out)))

    return _entities
