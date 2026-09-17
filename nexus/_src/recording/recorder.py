"""Recorder + RecordChannel: the generic mechanics of the observation seam.

A :class:`RecordChannel` is one component's per-tick buffer: a device ring buffer of ``width``-float
rows the component's capturable kernel writes into, plus the locked read-time D2H + decode the host
calls. The ring-buffer mechanics, a device counter advancing per graph replay, the wrap, and time =
counter × dt, are the same across components; only the row width + the row→object decode differ, so
each component supplies those when it registers its channel. :class:`Recorder` is the registry of
named channels handed to each recordable component: the single switch, ``None`` when not observing,
exactly as with ``Logger``.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Callable, Mapping

import numpy as np
import warp as wp


def leaf_keys(labels: list[str]) -> list[str]:
    """Friendly channel keys from full model labels, often Universal Scene Description (USD) prim paths
    such as ``/astro_max/Geometry/body_frd``: the leaf name after the last ``/`` when it's unique across
    the set, as in ``body_frd`` or ``rotor_1``, else the full label, so a leaf collision never silently
    aliases two entities. The ``sim.physics[...]`` keys are exactly these.
    """
    leaves = [str(lbl).rsplit("/", 1)[-1] for lbl in labels]
    counts = Counter(leaves)
    return [leaf if counts[leaf] == 1 else str(lbl) for lbl, leaf in zip(labels, leaves, strict=True)]


class RecordChannel:
    """One component's per-tick observation buffer.

    The component's capturable kernel writes a ``width``-element row each tick into :attr:`buf` at the
    device :attr:`counter`, which advances per graph replay, so the row index is correct under capture;
    the host reads via :meth:`latest` / :meth:`history` with one locked D2H and the channel's ``decode``.
    The host reconstructs per-row time from the counter × :attr:`dt`, because a kernel arg freezes at
    capture and the counter doesn't, so the writing kernel must stamp ``row[0] = dt × counter`` itself.
    ``dtype`` is the row element type, default ``float`` = f32; a Global Positioning System (GPS)
    measurement channel uses ``wp.float64`` so lat/lon survive, and the writing kernel's buffer arg
    must match.

    ``fields`` is the channel's declared row schema: ordered ``(name, n_columns)`` pairs covering the
    columns after the implicit time column 0, for example ``(("position", 3), ("quat_xyzw", 4), …)``. It's
    the single declaration everything downstream derives from: :meth:`history_arrays`, the vectorized
    per-quantity readback, and the Rerun debug time-series dump/tab tree. ``source`` names the
    registering implementation class, ``ImuSensor`` say: kind metadata for tools; the channel key stays
    the flat instance name, so redundant future instances are siblings, as in ``sensors/imu_bosch``.
    """

    def __init__(
        self,
        width: int,
        dt: float,
        decode: Callable[[np.ndarray], object],
        maxlen: int = 4096,
        dtype: type = float,
        fields: tuple[tuple[str, int], ...] | None = None,
        source: str | None = None,
    ):
        self.width = int(width)
        self.dt = float(dt)
        self.maxlen = int(maxlen)
        self.dtype = dtype
        self.fields = tuple((str(n), int(w)) for n, w in fields) if fields is not None else None
        if self.fields is not None and 1 + sum(w for _, w in self.fields) != self.width:
            raise ValueError(
                f"fields {self.fields} cover {1 + sum(w for _, w in self.fields)} columns "
                f"(incl. the implicit t) but width is {self.width}"
            )
        self.source = source
        self.buf = wp.zeros((maxlen, width), dtype=dtype)  # device ring buffer
        self.counter = wp.zeros(1, dtype=int)  # total rows; advances IN the graph
        self._decode = decode
        self._lock = threading.Lock()  # serialise the readback: one consistent (counter, buf) pair per reader

    def _snapshot(self) -> tuple[int, np.ndarray]:
        with self._lock:  # one consistent readback; .numpy() syncs, so it's the last completed tick's rows
            return int(self.counter.numpy()[0]), self.buf.numpy()

    def latest(self) -> object:
        """Return the most recently written row, decoded.

        Reads one row off the device, not the whole ring. That matters because this is the
        per-tick read: ``Sim.wait_until`` calls it every control tick for the sim clock, and an
        unbounded PX4 run sizes its ring at 30k rows, about 1.7 MB, so copying the ring here to
        return a single row cost ~0.14 ms per tick against a ~4 ms budget.

        Raises:
            RuntimeError: No row exists yet: sim not started or not stepped.
        """
        with self._lock:  # one consistent (counter, row) pair; .numpy() syncs
            c = int(self.counter.numpy()[0])
            if c == 0:
                raise RuntimeError("RecordChannel has no row yet (sim not started / not stepped)")
            row = self.buf[(c - 1) % self.maxlen].numpy()
        return self._decode(row)

    def history(self) -> list:
        """Return the buffered rows, oldest first and bounded by :attr:`maxlen`, each decoded."""
        return [self._decode(r) for r in self._history_rows()]

    def _history_rows(self) -> np.ndarray:
        """The buffered raw rows ``(N, width)``, oldest first: one locked D2H, ring unwrapped."""
        c, buf = self._snapshot()
        if c <= self.maxlen:
            return buf[:c]
        s = c % self.maxlen  # the ring wrapped: the oldest row is at c % maxlen
        return np.concatenate([buf[s:], buf[:s]])

    def history_arrays(self) -> dict[str, np.ndarray]:
        """The buffered history as one array per declared quantity, ``{"t": (N,), name: (N,) | (N, w)}``,
        oldest first, vectorized straight off the ring with no per-row decode and the channel's dtype.
        Width-1 quantities come back 1-D. ``N == 0`` before the first tick.

        Raises:
            ValueError: The channel declared no ``fields`` schema.
        """
        if self.fields is None:
            raise ValueError("channel declared no fields schema (register with fields=...)")
        rows = self._history_rows()
        out = {"t": rows[:, 0].copy()}
        col = 1
        for name, w in self.fields:
            out[name] = rows[:, col].copy() if w == 1 else rows[:, col : col + w].copy()
            col += w
        return out


class Recorder:
    """The observation sink handed to each recordable component: the single switch, ``None`` when off.

    Owns the named channels; a component registers its channel via :meth:`channel` in its
    ``set_recorder`` and the host reads them via :attr:`channels`, for example ``recorder.channels["state"]``.
    ``base_body`` is the one channel with a name of its own: physics points it at the vehicle's base
    body when it registers, and the flown path is drawn from it.
    """

    def __init__(self, dt: float, maxlen: int = 4096):
        self._dt = float(dt)
        self._maxlen = int(maxlen)
        self.channels: dict[str, RecordChannel] = {}
        self.base_body: RecordChannel | None = None  # the base body's channel: the flown path

    def dump_to(self, sink) -> None:
        """Dump every channel's ring into the recording at teardown, through the Rerun adapter.

        Args:
            sink: The Logger the run records to.
        """
        from .rerun_adapter import dump

        dump(self.channels, sink, base_body=self.base_body)

    def channel(
        self,
        name: str,
        *,
        width: int,
        decode: Callable[[np.ndarray], object],
        dtype: type = float,
        fields: tuple[tuple[str, int], ...] | None = None,
        source: str | None = None,
    ) -> RecordChannel:
        """Register a component's channel: a ``width``-element row buffer with its decode and its
        declared ``fields`` schema / ``source`` metadata; see :class:`RecordChannel`.

        Idempotent for a re-registration with the same width/dtype/fields/source, which returns the
        existing channel, but a duplicate name with a different shape raises: two components must never
        silently share one ring, for example two Inertial Measurement Unit (IMU) instances both named
        ``imu``; give them distinct instance names instead: ``imu_bosch`` / ``imu_murata``.

        ``dtype`` is the row element type: default ``float`` = f32, and a GPS channel passes ``wp.float64``.
        """
        ch = self.channels.get(name)
        if ch is not None:
            declared = tuple((str(n), int(w)) for n, w in fields) if fields is not None else None
            if (ch.width, ch.dtype, ch.fields, ch.source) != (int(width), dtype, declared, source):
                raise ValueError(
                    f"channel {name!r} is already registered with a different shape "
                    f"(width={ch.width}, dtype={ch.dtype}, fields={ch.fields}, source={ch.source}): "
                    "distinct components need distinct instance names"
                )
            return ch
        ch = RecordChannel(width, self._dt, decode, self._maxlen, dtype, fields=fields, source=source)
        self.channels[name] = ch
        return ch


class ChannelMap(Mapping):
    """A read-only, name-keyed view over a subset of a Recorder's channels: the host-facing surface.

    ``Sim.physics`` and ``Sim.sensors`` are ``ChannelMap``s over the flat ``Recorder.channels`` dict,
    selected by component-kind key prefix and addressed by the bare entity name:
    ``sim.physics["rotor_1_joint"]`` resolves ``channels["physics/joint/rotor_1_joint"]``, and
    ``Sim.physics`` spans both ``physics/body/`` and ``physics/joint/``. A lookup returns the
    :class:`RecordChannel`, so the caller picks ``.latest()`` / ``.history()``.
    """

    def __init__(self, channels: dict[str, RecordChannel], prefixes: tuple[str, ...]):
        self._channels = channels
        self._prefixes = prefixes

    def _resolve(self, name: str) -> str:
        hits = [p + name for p in self._prefixes if (p + name) in self._channels]
        if not hits:
            raise KeyError(name)
        if len(hits) > 1:  # the same bare name under two namespaces, say a body and a joint: ambiguous
            raise KeyError(f"ambiguous entity name {name!r}: matches {hits}")
        return hits[0]

    def __getitem__(self, name: str) -> RecordChannel:
        return self._channels[self._resolve(name)]

    def __iter__(self):
        for key in self._channels:
            for p in self._prefixes:
                if key.startswith(p):
                    yield key[len(p) :]

    def __len__(self) -> int:
        return sum(1 for _ in self)
