"""The serial link the run owns between two peers: a companion computer and the flight controller talk over a
serial line on the vehicle, and in the run that wire is two pseudo-terminals joined by a relay this process holds.

A pseudo-terminal's master has no path, so two processes that each open a path need a pair each, and the run
copies bytes between the two masters. The link keeps its own handle on each slave in raw mode, so a peer that
opens and closes its end, or hasn't opened it yet, never hangs the relay up, and no line discipline touches
the bytes. The run makes every link before any peer starts, hands each peer its end's path, and stops the links
with the peers at the run's end, so the link speaks the teardown half of the peer contract: ``stop``, ``alive``
and ``name``.
"""

from __future__ import annotations

import os
import select
import threading
import tty

_CHUNK = 65536


class SerialLink:
    """One serial link, the run's cable for the line ``line``, the Universal Scene Description (USD) name of the
    relationship that wires it, such as ``/vehicle/Companion.nexus:serial:autopilot``.

    ``ends`` holds the two paths, each a ``/dev/pts/N`` a peer opens as a serial device: the first for the peer
    whose prim wires the line, the second for the peer at its other end. The relay runs from the moment the run
    makes the link, so a peer that starts first can write before the other opens its end; the bytes wait in the
    other end's buffer.

    Args:
        line: The USD name of the line the link stands for, which names the relay's thread.
    """

    def __init__(self, line: str) -> None:
        self.name = f"serial link {line}"
        self._masters = []
        self._slaves = []
        ends = []
        for _ in range(2):
            master, slave = os.openpty()
            tty.setraw(slave)
            os.set_blocking(master, False)  # a full far end never blocks the relay past a stop
            self._masters.append(master)
            self._slaves.append(slave)
            ends.append(os.ttyname(slave))
        self.ends: tuple[str, str] = (ends[0], ends[1])
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._relay, name=self.name, daemon=True)
        self._thread.start()

    def start(self) -> None:
        """Nothing: the link relays from the moment the run makes it."""

    def stop(self) -> None:
        """End the relay and close both ends, so neither path is a device any more. Idempotent."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
        for fd in [*self._masters, *self._slaves]:
            try:
                os.close(fd)
            except OSError:
                pass
        self._masters, self._slaves = [], []

    def alive(self) -> bool:
        """Whether the relay still runs."""
        return self._thread.is_alive()

    def artifacts(self) -> dict:
        """No file: the link writes none."""
        return {}

    def _relay(self) -> None:
        other = dict(zip(self._masters, reversed(self._masters), strict=True))
        while not self._stop.is_set():
            ready, _, _ = select.select(self._masters, [], [], 0.1)
            for fd in ready:
                try:
                    data = os.read(fd, _CHUNK)
                except BlockingIOError:
                    continue
                except OSError:
                    return
                self._write(other[fd], data)

    def _write(self, fd: int, data: bytes) -> None:
        """Write `data` whole to the master `fd`, waiting while the far end's buffer is full, until a stop."""
        while data and not self._stop.is_set():
            try:
                data = data[os.write(fd, data) :]
            except BlockingIOError:
                select.select([], [fd], [], 0.1)
