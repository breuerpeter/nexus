"""The folder that holds one run's files: its recording, ``nexus.rrd``, and each peer's logs, such as
PX4's console, ``px4.log``, and its ULog, and Kit's console, ``kit.log``.

The build picks one folder per run and hands it to each writer. A writer creates it when it writes
the first file, so a run that writes nothing leaves no folder.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

_taken: set[str] = set()  # the names this process gave out, so two runs in one second take two folders


def run_folder() -> Path:
    """A new folder for one run, ``~/.cache/nexus/logs/<stamp>-<pid>``, with ``-<n>`` after it for a
    later run of this process in the same second. The folder isn't created.
    """
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    n = 1
    while (taken := name if n == 1 else f"{name}-{n}") in _taken:
        n += 1
    _taken.add(taken)
    return Path("~/.cache/nexus/logs").expanduser() / taken
