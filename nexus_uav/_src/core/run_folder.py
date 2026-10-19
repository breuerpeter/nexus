"""The folder that holds one run's files: its recording, ``nexus.rrd``, and each peer's files under a folder
named for the peer, such as PX4's console, ``px4_sitl/console.log``, and its ULog beside it, and Kit's console,
``kit/console.log``.

The build picks one folder per run and hands it to each writer: the folder the caller names, or a new one
under ``~/.cache/nexus/logs``. Picking the new one creates it, so no two runs share it, and a run that writes
nothing leaves an empty one. The build refuses a named folder that holds anything, so a run never mixes its
files with an earlier run's. The caller owns a folder it names: two runs that name one folder at once both
find it empty.
"""

from __future__ import annotations

import os
import time
from pathlib import Path


def run_folder(path: str | os.PathLike | None = None) -> Path:
    """The folder for one run: ``path`` when given, else a new folder ``~/.cache/nexus/logs/<stamp>-<pid>``.

    The call creates ``path`` and its parents when missing, and takes an empty ``path`` as it stands. The
    new folder takes ``-<n>`` after its name when an earlier run holds that name: its creation fails on a
    name that exists, so a reused process id or a clock that steps back never hands a run an earlier run's
    folder.

    Raises:
        FileExistsError: ``path`` holds a file or a folder already, or is a file.
    """
    if path is not None:
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        if any(folder.iterdir()):
            raise FileExistsError(f"the run's folder {folder} holds files already: name a new or empty folder")
        return folder
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
