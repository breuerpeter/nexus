"""The peer contract: a process the run starts and then speaks to over a real protocol.

A peer is only ever a process, a container, a virtual machine or a bare process. The component
that speaks to it, the loop face, is separate from it, so a run can hold the component with no
peer: an autopilot started elsewhere, or a real one on a bench. The run chooses the peer's
realization, managed, a process the build starts through this contract and the run stops, or
external, an address the run attaches to, and the build makes the component the same way for both.

The contract is narrow on purpose: start, stop, and whether it's alive. No watchdog, no restart
policy and no ordering beyond what a peer's own start does. A peer's death reaches the loop through
its link, so the loop needs nothing more from the peer itself. The run owns every address a peer
uses and hands the peer its ports, so two runs on one machine don't collide.

The PX4 Software In The Loop (SITL) container, :class:`nexus._src.vehicle.controllers.px4.sitl.Px4Sitl`,
is the first peer.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


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


__all__ = ["Peer"]
