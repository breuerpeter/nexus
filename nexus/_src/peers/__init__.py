"""The peer contract: a process the run starts and then speaks to over a real protocol.

A peer is a process outside the loop's process: a container, a virtual machine or a bare process.
A component, a class in the loop's process, speaks to it over a link, so a run can hold the
component with no peer: an autopilot started elsewhere, or a real one on a bench. The vehicle's
Universal Scene Description (USD) file declares its peers, and the build starts each one it
declares through this contract, and the run stops it. A run's override layer drops a declaration to
attach to a process started elsewhere, and the build makes the component the same way for both.

The contract is narrow on purpose: start, stop, and whether it's alive. No watchdog, no restart
policy and no ordering beyond what a peer's own start does. A peer's death reaches the loop through
its link, so the loop needs nothing more from the peer itself. The run owns every address a peer
uses and hands the peer its ports, so two runs on one machine don't collide.

Each peer lives in one folder, ``peers/<name>/``, with everything that belongs to its process.
Its host-side modules sit at the top of the folder, and what crosses into the process sits in a
subfolder with a fixed name: ``image/``, a docker build context the image tag hashes, and
``peer-src/``, what the container mounts read-only. The hyphen keeps ``peer-src/`` out of every
dotted import, so no host module loads the peer's program. This folder itself holds the contract
and :mod:`.containers`, the docker runner every container peer shares. The peers are the PX4
Software In The Loop (SITL) container, :mod:`.px4_sitl`, and the Kit render peer, :mod:`.kit`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

LABEL = "nexus.peer"
"""The container label that marks a peer by its kind, ``px4`` or ``kit``, so a start or a runner can
find a leftover one."""


@runtime_checkable
class Peer(Protocol):
    """A process the run starts, stops and can ask after.

    ``start`` blocks until the process is running, or raises naming why it couldn't start. ``stop``
    is idempotent: a peer that never started, or stopped already, stays stopped. ``alive`` says
    whether the process still runs; a peer that exited on its own reads dead. A peer that writes
    files a reader wants after the run, a console log, names them in ``artifacts``, which is
    optional.
    """

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def alive(self) -> bool: ...


__all__ = ["LABEL", "Peer"]
