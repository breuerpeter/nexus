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
uses and hands the peer its ports, so two runs on one machine don't collide. The builder builds a
link's end inside the run from those addresses. A link's end outside the run, such as the client a
script opens on PX4's offboard link, reads its address from the run's port map, ``sim.ports``.

Each peer lives in one folder, ``peers/<name>/``, with everything that belongs to its process.
Its host-side modules sit at the top of the folder, and what crosses into the process sits in a
subfolder with a fixed name: ``image/``, a docker build context the image tag hashes, and
``peer-src/``, what the container mounts read-only. The hyphen keeps ``peer-src/`` out of every
dotted import, so no host module loads the peer's program. Each peer pins what it runs, a commit
or an image, in ``pin.ref`` at the top of its folder, and its runner offers ``prepare()``, the cold work the
peer's first start does, a fetch and a build, or an image pull, for a caller that wants it outside a run's
budget: CI before a flight or a test, or a machine's first setup. Every peer's files live under a folder named for
the peer in the run's folder, its console as ``console.log`` there, and what a peer keeps across runs, a tree
it built or a shader cache, lives under ``~/.cache/nexus/<name>/``, so CI caches a peer's box by that one
rule. This folder itself holds the contract,
:mod:`.containers`, the container peer every container peer subclasses and the image a peer builds from a
folder, both public, :mod:`.pins`, which finds every peer's pin the same way, :mod:`.instance`, which claims
the instance a run numbers its peers' ports from, :mod:`.files`, the files a run's peers read, and
:mod:`.serial`, the serial link the run makes between two peers. The peers are the
PX4 Software In The Loop (SITL) container, :mod:`.px4_sitl`, and the Kit render peer, :mod:`.kit`.
"""

from __future__ import annotations

from typing import ClassVar, Literal, Protocol, runtime_checkable

LABEL = "nexus.peer"
"""The container label that marks a peer by its kind, its ``name``, ``px4_sitl`` or ``kit``, so a start or a
runner can find a leftover one."""


CLOCKS = ("stepped", "on_request", "free")
"""The three ways a peer keeps time, the values of a peer class's ``clock``."""


@runtime_checkable
class Peer(Protocol):
    """A process the run starts, stops and can ask after.

    ``start`` blocks until the process is running, or raises naming why it couldn't start. ``stop``
    is idempotent: a peer that never started, or stopped already, stays stopped. ``alive`` says
    whether the process still runs; a peer that exited on its own reads dead, and a run waiting on
    it fails at once. A peer that writes files a reader wants after the run, a console log, names
    them in ``artifacts``, which is optional. A peer that can tell its own error lines from the rest
    of its console, such as the line PX4 prints for an airframe its tree lacks, returns them from
    ``errors``, which is optional too, and a run that fails quotes them in its ``RunError``.

    ``clock`` states how the peer keeps time, one of three ways. ``stepped``: a component steps it each
    tick, as PX4 SITL runs on the sim time the lockstep link carries. ``on_request``: it answers a request
    for a given sim time, as the Kit peer renders the frame a tick asks for. ``free``: it runs on its own
    wall clock, as a companion computer under an emulator does. Sim time and a free-running peer's clock
    agree only while the run paces at a real-time factor of 1, so the build fails a run that declares one
    at any other factor, before any peer starts, and the loop warns when the live factor falls below 0.95.

    A peer's schema can declare a serial line, a relationship ``nexus:serial:<line>`` whose target is the prim of
    the peer at the line's other end, as a companion computer's line to the flight controller. The run owns the
    link, :mod:`.serial`: it makes one per line before any peer starts and hands each of the two peers the path of
    its end in the keyword ``serial``, the declaring peer by the line's name and the other by the declaring prim's
    name. A peer with an end takes ``serial`` beside the run's other values and opens the path it names.

    A peer's schema can declare an input, a relationship ``nexus:inputs:<signal>`` whose targets are the prims
    whose components write that signal, one or more, as a camera's USB cable plugs into a companion computer. The
    loop end of that link is the run's: once per tick, after the writers, it hands the peer each writer's value
    through ``take(signal, writer, value)``, with the writer's instance name, which a peer with an input states.
    The peer decides what it makes of a value and how it carries it. A value with no fixed width stays on its
    signal until the next, so a peer sends a frame once, by its time. A ``take`` that raises fails the run, and
    the error names the peer.
    """

    clock: ClassVar[Literal["stepped", "on_request", "free"]]

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def alive(self) -> bool: ...


__all__ = ["CLOCKS", "LABEL", "Peer"]
