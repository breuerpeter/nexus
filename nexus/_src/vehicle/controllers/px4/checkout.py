"""How the PX4 controller reaches its PX4 tree: the pin it ships, the pin a project keeps beside its
catalog, ``$PX4_DIR`` as an override, and the fetch of the pinned commit into a folder it owns.
"""

from __future__ import annotations

from pathlib import Path


def fetch(dest: Path, *, url: str, sha: str) -> None: ...


def pin(catalog: Path | None = None): ...


def tree(catalog: Path | None = None) -> Path: ...
