"""The fake Kit render peer: a stand-in that speaks the render link and starts no container."""

from __future__ import annotations


class KitFake:
    def __init__(self, *, error_after: int | None = None, error: str = "") -> None: ...
