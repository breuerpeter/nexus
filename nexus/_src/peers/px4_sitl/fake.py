"""The fake PX4 Software In The Loop (SITL) peer: a stand-in that speaks the Hardware In The Loop (HIL) link and
starts no process.
"""

from __future__ import annotations


class Px4Fake:
    def __init__(self, *, port: int, system_id: int) -> None: ...
