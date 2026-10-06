The render link's host end: `link.py` is the renderer the loop drives and the link the RTX sensors
ride. The Kit peer it speaks to lives in `nexus_sim/_src/peers/kit/`. Read its `CLAUDE.md` before
changing the peer, its program or the link's definition.

- Kit renders one update behind, so a frame reply carries the request before it: the first reply
  carries nothing, and the close renders the last request and returns its frame with `closed`.
- `link.py` defines no message of its own: it runs `peers/kit/peer-src/link.py` by path and takes
  the framing and the message names from it.
