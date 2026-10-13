"""The folder that holds one run's files: its recording, ``nexus.rrd``, and each peer's logs, such as
PX4's console, ``px4.log``, and its ULog, and Kit's console, ``kit.log``.

The build picks one folder per run and hands it to each writer. Picking it creates it, so no two runs
share a folder, and a run that writes nothing leaves an empty one.
"""

from __future__ import annotations

import os
import time
from pathlib import Path


def run_folder() -> Path:
    """A new folder for one run, ``~/.cache/nexus/logs/<stamp>-<pid>``, with ``-<n>`` after it when
    an earlier run holds that name. This call creates the folder, and the creation fails on a name that
    exists, so a reused process id or a clock that steps back never hands a run an earlier run's folder.
    """
    logs = Path("~/.cache/nexus/logs").expanduser()
    logs.mkdir(parents=True, exist_ok=True)
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    n = 1
    while True:
        folder = logs / (name if n == 1 else f"{name}-{n}")
        try:
            folder.mkdir()
        except FileExistsError:
            n += 1
            continue
        return folder
