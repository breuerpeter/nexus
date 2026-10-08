"""The run's port map: each link that leaves the run, by name, to the address its client opens.

The run owns every address. The builder builds a link's end inside the run from those addresses,
the Hardware In The Loop (HIL) server the PX4 controller listens on, and names the links whose
other end sits outside the run here: the offboard link a script opens a client on. A link the run
could name but has nothing behind, the offboard link of a fake PX4 that answers it not, stays out
of the map with the reason, so a lookup fails at once naming it, and a script opens no client on a
link nothing answers.
"""

from __future__ import annotations

from collections.abc import Mapping


class PortMap(dict):
    """The links that leave the run, by name, each to the address its client opens: a mapping of
    plain entries, ``{"protocol": "udp", "port": 14540, "system_id": 1}`` for PX4's offboard link.

    Args:
        links: The links the run names, by name, each to its entry.
        missing: The links the run could have named but keeps out of the map, by name, each to the
            reason, which a lookup of it raises with.
    """

    def __init__(self, links: Mapping[str, dict] | None = None, *, missing: Mapping[str, str] | None = None):
        super().__init__(links or {})
        self._missing = dict(missing or {})

    def __missing__(self, key: str):
        if key in self._missing:
            raise KeyError(f"no {key!r} link in this run's port map: {self._missing[key]}")
        raise KeyError(f"no {key!r} link in this run's port map; the links are {sorted(self)}")
