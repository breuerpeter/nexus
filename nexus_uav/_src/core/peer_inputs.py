"""The loop end of a declared peer's inputs: the link the vehicle wires from each writer's output to the peer.

A peer's schema declares each input as a relationship ``nexus:inputs:<signal>``, and the vehicle targets the prims
whose components write that signal, one or more, as each camera on the drone plugs into the companion computer.
The builder makes one :class:`PeerInputs` per wired peer, a component in the ``link`` role, whose one host stage
runs after every writer of the tick: it reads each input as a list, one value per targeted writer, and hands each
to the peer's ``take`` with the writer's name. ``take`` is the peer's end of the link, and the peer decides what
it makes of a value and how it carries it. The wire carries each value once, as a camera's cable carries each
frame once: the stage hands a host value on the write that produced it, so a frame reaches the peer at the
camera's rate and not at the tick's, and a device value every tick, since a kernel writes it every tick. A host
signal its writer hasn't written yet, a camera before its first frame, waits. A ``take`` that raises fails the
run with a ``RunError`` naming the peer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .errors import RunError
from .interfaces import Component, Stage
from .signals import HostBuffer, Signal


class PeerInputs(Component):
    """The link's loop end for one declared peer: a component named for the peer, in the ``link`` role.

    Args:
        peer: The peer, whose ``take(signal, writer, value)`` takes each value and whose ``name`` names the stage.
        inputs: Each wired input's name to the outputs its targeted writers declare, in the vehicle's order, from
            which the stage declares a list of their type and shape.
    """

    def __init__(self, peer, inputs: Mapping[str, Sequence[Signal]]):
        self.peer = peer
        self.name = str(getattr(peer, "name", None) or type(peer).__name__)
        self._inputs = tuple(
            Signal(name, list[outputs[0].type], shape=outputs[0].shape, default=()) for name, outputs in inputs.items()
        )
        # The writes of each host buffer whose value the peer has, by input and writer.
        self._handed: dict[tuple[str, str], int] = {}

    def stages(self) -> list[Stage]:
        """One host stage, named for the peer, which reads every wired input and hands each value to the peer."""
        return [Stage(self.name, "host", self._take, reads=self._inputs)]

    def _take(self, tick) -> None:
        for signal in self._inputs:
            buffers = signal.buffer if signal.buffer else ()
            for writer, buffer, value in zip(signal.writers, buffers, signal.read(), strict=True):
                if isinstance(buffer, HostBuffer):
                    # A host value once, on the write that produced it; none before the first write.
                    if buffer.writes == self._handed.get((signal.name, writer), 0):
                        continue
                    self._handed[signal.name, writer] = buffer.writes
                try:
                    self.peer.take(signal.name, writer, value)
                except Exception as exc:
                    raise RunError(f"the {self.name} peer's link refused {writer}'s {signal.name}: {exc}") from exc
