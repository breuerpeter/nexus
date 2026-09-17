"""SeedTree: the one seeded Random Number Generator (RNG) tree.

Two consumer APIs off one root seed:

* :meth:`stream`: a Python ``random.Random``, the legacy host path; v1 handed every consumer the
  *same* stream for bridge parity.
* :meth:`seed_for`: a deterministic per-consumer **integer** seed for the Warp RNG, ``wp.rand_init``.
  This is the "independent named sub-stream per consumer" the v1 docstring deferred: it
  intentionally gives each Warp sensor its own reproducible noise field, a seed plus a per-tick counter,
  re-baselining determinism off the bridge's single shared sequence onto the device RNG, the
  prerequisite for capturable / tape-able sensors.
"""

from __future__ import annotations

import hashlib
import random


class SeedTree:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self._root = random.Random(seed)

    def stream(self, name: str) -> random.Random:
        # Legacy host path: shared stream -> the same draw order as the bridge.
        return self._root

    def seed_for(self, name: str) -> int:
        """A deterministic 32-bit seed for ``name``, derived from the root seed. Stable across runs:
        ``hashlib``, not Python ``hash()``, which salts its output. Feeds ``wp.rand_init`` in Warp sensors.
        """
        digest = hashlib.sha256(f"{self.seed}:{name}".encode()).digest()
        return int.from_bytes(digest[:4], "little")
