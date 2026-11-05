"""The cache root: the one folder that holds every path nexus keeps across runs.

The root is ``$NEXUS_CACHE_DIR`` when set, else ``$XDG_CACHE_HOME/nexus``, else ``~/.cache/nexus``, the
order uv and pip resolve their own caches in. Each path joins a name to the root: the PX4 tree and store
under ``px4_sitl/``, Kit's caches under ``kit/``, the run folders under ``logs/``. A caller that runs nexus
inside a container, and starts the peers' containers through the host's daemon, sets ``NEXUS_CACHE_DIR`` to
a folder bound at its host path, so every bind-mount source exists on the host too.
"""

from __future__ import annotations

import os
from pathlib import Path


def cache_root() -> Path:
    """The cache root, read from the environment on each call, so a variable set after import still holds."""
    if root := os.environ.get("NEXUS_CACHE_DIR"):
        return Path(root).expanduser()
    if xdg := os.environ.get("XDG_CACHE_HOME"):
        return Path(xdg).expanduser() / "nexus"
    return Path.home() / ".cache" / "nexus"
