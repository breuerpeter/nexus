The Kit render link's host end: `link.py` speaks to the Kit peer in `nexus_uav/_src/peers/kit/`, which no
RTX sensor requires any more, and it leaves with that peer. The RTX sensors render in the loop's process,
`nexus_uav/_src/vehicle/sensors/rtx/`. Read the Kit peer's `CLAUDE.md` before changing the peer, its program
or the link's definition.

- Kit renders one update behind, so a frame reply carries the request before it: the first reply
  carries nothing, and the close renders the last request and returns its frame with `closed`.
- `link.py` defines no message of its own: it runs `peers/kit/peer-src/link.py` by path and takes
  the framing and the message names from it.
