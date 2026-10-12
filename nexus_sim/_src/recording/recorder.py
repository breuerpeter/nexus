"""Recorder and History: every row a run records, with a fixed footprint on the device.

A :class:`History` is one signal's rows over a run. A device kernel writes one row per tick into its staging
buffers, the row's values and the device clock's sim time, at the slot the device counter names, so the write joins the
captured graph with no host readback. The loop commits one row per tick on the host, and each time the
staging buffers fill they drain: the rows copy onto the end of the host blocks, which grow with the run,
and the kernel writes slot 0 again. So device memory stays at ``staging`` rows per history and the Recorder
drops no row. The host reads a history on demand: ``latest`` reads the newest row off the device, as one
record, ``history`` every row, the blocks then the rows still staged, as one structured array.

The :class:`Recorder` holds the histories by key and records everything itself: :meth:`tap` registers one
history per device signal a component writes, under the writer's path then the signal's name, the plant's
bodies and joints among them, :meth:`record` snapshots them all in the record stage, and :meth:`commit`
counts the tick's row and drains the staging buffers when full. No component states a recording member.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import warp as wp

from nexus_sim._src.core.signals import Signal

STAGING = 4096  # rows a staging buffer holds: about 16 s at 250 Hz, the most a SIGKILL can lose


@wp.kernel
def record_signal(
    src: wp.array(dtype=Any),
    time: wp.array(dtype=wp.float64),
    staging: int,
    values: wp.array2d(dtype=Any),
    times: wp.array(dtype=wp.float64),
    counter: wp.array(dtype=int),
):
    """Copy a signal's buffer, flat, into the row the device counter names, and stamp the row with the tick's
    sim time, ``time``. One generic kernel serves every signal type: a struct, a value type or a row of floats.
    """
    c = counter[0]  # rows written so far; advances per graph replay, on the device, not a frozen kernel arg
    s = c % staging  # the staging slot: the host drains the buffer before the slot comes round again
    times[s] = time[0]
    for k in range(src.shape[0]):
        values[s, k] = src[k]
    counter[0] = c + 1


class History:
    """One signal's rows over a run.

    A row is ``width`` values of ``dtype``, a Warp element type, and its sim time. The writing kernel takes
    :attr:`values`, the device staging buffer of ``staging`` rows, :attr:`times`, their sim times,
    :attr:`counter`, the rows written so far, which advances on the device, and the device clock's time, which
    stamps the row: a kernel argument freezes at capture, and the counter and the clock don't. The host
    reconstructs nothing.

    ``fields`` is the declared row schema: ordered ``(name, n_values)`` pairs that cover the row, a struct's
    fields for a struct signal, such as ``(("position", 3), ("orientation", 4), …)`` for a body, or one
    quantity named after a value-typed signal. It's the one declaration everything downstream derives from:
    the records a row reads as, :meth:`arrays`, the per-quantity readback, and the series the Logger writes.
    ``source`` names the writing class, ``ImuSensor`` say: kind metadata for tools. ``path`` is the writer's
    path below the process root and ``row`` the signal's name under it.
    """

    def __init__(
        self,
        width: int,
        fields: tuple[tuple[str, int], ...],
        *,
        staging: int = STAGING,
        dtype: Any = float,
        source: str | None = None,
        path: str | None = None,
        row: str | None = None,
    ):
        self.width = int(width)
        self.staging = int(staging)
        self.dtype = dtype
        self.fields = tuple((str(n), int(w)) for n, w in fields)
        self.source = source
        self.path = path
        self.row = row
        self.values = wp.zeros((self.staging, self.width), dtype=dtype)  # the staged rows' values
        self.times = wp.zeros(self.staging, dtype=wp.float64)  # the staged rows' sim times
        self.counter = wp.zeros(1, dtype=int)  # rows written so far; advances IN the graph
        self.rows = 0  # rows the loop has committed: the host's count, which decides the drains
        self._blocks: list[tuple[np.ndarray, np.ndarray]] = []  # the drained (times, values), oldest first
        self._lock = threading.Lock()  # serialise the readback: one consistent (counter, buffers) per reader
        if not self._structured and sum(w for _, w in self.fields) != self._columns:
            raise ValueError(
                f"fields {self.fields} cover {sum(w for _, w in self.fields)} values but width is {self.width}"
            )

    @property
    def _structured(self) -> bool:
        """Whether a value is a Warp struct, whose fields name the quantities."""
        return wp.types.type_is_struct(self.dtype)

    @property
    def _columns(self) -> int:
        """The floats one row holds once flattened: ``width`` values of ``length`` elements each."""
        return self.width * (1 if self._structured else int(wp.types.type_size(self.dtype)))

    @property
    def _drained(self) -> int:
        return len(self._blocks) * self.staging

    def commit(self) -> bool:
        """Count the row the record stage wrote this tick, and drain the staging buffers when full.

        Returns:
            Whether a drain happened: the last ``staging`` rows are now a host block.
        """
        self.rows += 1
        if self.rows - self._drained < self.staging:
            return False
        with self._lock:
            self._blocks.append((self.times.numpy().copy(), self.values.numpy().copy()))  # one copy per drain
        return True

    def _snapshot(self) -> tuple[int, np.ndarray, np.ndarray]:
        with self._lock:  # one consistent readback; .numpy() syncs, so it's the last completed tick's rows
            return int(self.counter.numpy()[0]), self.times.numpy(), self.values.numpy()

    def latest(self) -> np.void:
        """Return the most recently written row as one record: ``t``, the tick's sim time, then each declared
        quantity, ``row["position"]``.

        Reads one row off the device, not the whole staging buffer. That matters because this is the
        per-tick read: a script's predicate between steps.

        Raises:
            RuntimeError: No row exists yet: sim not started or not stepped.
        """
        with self._lock:
            c = int(self.counter.numpy()[0])
            if c == 0:
                raise RuntimeError("History has no row yet (sim not started / not stepped)")
            slot = (c - 1) % self.staging
            t, row = self.times[slot : slot + 1].numpy(), self.values[slot : slot + 1].numpy()
        return self._records(t, row)[0]

    def history(self) -> np.ndarray:
        """Return every row, oldest first, as one structured array: ``t`` then each declared quantity, with the
        value's own type, ``rows["position"][:, 2]``.
        """
        return self._records(*self._rows(0, None))

    def _records(self, times: np.ndarray, values: np.ndarray) -> np.ndarray:
        """The rows as one structured array: ``t`` then each declared quantity, with the value's own type."""
        split = self._split(values)
        dtype = [("t", np.float64)] + [(name, split[name].dtype, split[name].shape[1:]) for name, _ in self.fields]
        out = np.empty(len(times), dtype=dtype)
        out["t"] = times
        for name, _ in self.fields:
            out[name] = split[name]
        return out

    def _rows(self, start: int, stop: int | None) -> tuple[np.ndarray, np.ndarray]:
        """The raw rows ``[start, stop)`` as ``(times, values)``, oldest first: the host blocks that overlap
        the range, then the rows still on the device, read with one locked device-to-host copy.
        """
        c, times, values = self._snapshot()
        drained = self._drained
        stop = c if stop is None else min(int(stop), c)
        parts = []
        for i, (bt, bv) in enumerate(self._blocks):
            lo, hi = i * self.staging, (i + 1) * self.staging
            if hi > start and lo < stop:
                parts.append(
                    (bt[max(start, lo) - lo : min(stop, hi) - lo], bv[max(start, lo) - lo : min(stop, hi) - lo])
                )
        if stop > drained:
            parts.append(
                (
                    times[max(start, drained) - drained : stop - drained],
                    values[max(start, drained) - drained : stop - drained],
                )
            )
        if not parts:
            return np.empty(0, dtype=np.float64), np.empty((0, *values.shape[1:]), dtype=values.dtype)
        return np.concatenate([t for t, _ in parts]), np.concatenate([v for _, v in parts])

    def _split(self, values: np.ndarray) -> dict[str, np.ndarray]:
        """The rows' values as one array per declared quantity, ``(N,)`` for a width-1 quantity, else
        ``(N, w, …)``: a struct's fields, or the declared slices of a flat row.
        """
        n = len(values)
        out = {}
        if self._structured:
            flat = values.reshape(n, self.width)
            for name, _ in self.fields:
                field = flat[name]  # (N, width, *shape)
                out[name] = field[:, 0] if self.width == 1 else field
        else:
            flat = values.reshape(n, self._columns)
            col = 0
            for name, w in self.fields:
                out[name] = flat[:, col].copy() if w == 1 else flat[:, col : col + w].copy()
                col += w
        return out

    def arrays(self, start: int = 0, stop: int | None = None) -> dict[str, np.ndarray]:
        """The rows ``[start, stop)`` as one array per declared quantity, ``{"t": (N,), name: (N,) |
        (N, w)}``, oldest first, straight off the rows with no per-row decode and the value's own type.
        Width-1 quantities come back 1-D. ``N == 0`` before the first tick.
        """
        times, values = self._rows(start, stop)
        return {"t": times.copy(), **self._split(values)}

    def history_arrays(self) -> dict[str, np.ndarray]:
        """Every row as one array per declared quantity, as :meth:`arrays` over the whole run."""
        return self.arrays()


def _quantities(signal: Signal, width: int) -> tuple[tuple[str, int], ...]:
    """The quantities a signal's type declares: a struct's fields, each with its number of values, or one
    quantity named after the signal, of the flat buffer's width.
    """
    if wp.types.type_is_struct(signal.type):
        dtype = np.dtype(signal.type.numpy_dtype())
        return tuple((name, int(np.prod(dtype[name].shape, dtype=int))) for name in dtype.names)
    return ((signal.name, width * int(wp.types.type_size(signal.type))),)


class Recorder:
    """The histories of a run, by key, and the record taps.

    A history's key is its writer's path below the process root, the path its rows take in a recording,
    then its signal's name: ``vehicle/body/<label>`` and ``vehicle/joints/<label>`` for the plant's outputs,
    which the physics writes under its path ``vehicle``, and ``vehicle/sensors/imu/imu`` for a sensor's
    sample, each of which :meth:`tap` registers.
    """

    def __init__(self, staging: int = STAGING):
        self.staging = int(staging)
        self.histories: dict[str, History] = {}
        self.rows = 0  # rows committed so far, the same for every history
        self._taps: list[tuple[History, wp.array]] = []

    def tap(self, path: str, signal: Signal, *, source: str | None = None) -> History:
        """Register the history of a device signal a component writes, keyed ``<path>/<signal>``, and tap its
        buffer: each :meth:`record` copies what the signal holds into the history. The rows read as records
        of ``t`` then the quantities the signal's type declares, with the type's own dtypes.

        Args:
            path: The writer's path below the process root, such as ``vehicle/sensors/imu``.
            signal: The signal, wired: its buffer exists.
            source: The writing class's name, kind metadata for tools.

        Raises:
            ValueError: The key is already registered.
        """
        key = f"{path}/{signal.name}"
        if key in self.histories:
            raise ValueError(f"history {key!r} is already registered: one component writes {signal.name!r} once")
        src = signal.buffer if signal.buffer.ndim == 1 else signal.buffer.flatten()
        width = int(src.shape[0])
        h = History(
            width,
            _quantities(signal, width),
            staging=self.staging,
            dtype=signal.type,
            source=source,
            path=path,
            row=signal.name,
        )
        self.histories[key] = h
        self._taps.append((h, src))
        return h

    def record(self, time: wp.array) -> None:
        """The record taps: snapshot every tapped signal into its history, device-only, so the launches join
        the captured graph. The loop's record stage runs this each tick.

        Args:
            time: The tick's sim time on the device, the device clock's signal ``time``, which stamps each row.
        """
        for h, src in self._taps:
            wp.launch(record_signal, dim=1, inputs=(src, time, h.staging, h.values, h.times, h.counter))

    def commit(self) -> tuple[int, int] | None:
        """Count the row every history gained this tick, and drain the staging buffers when full.

        Returns:
            The drained block as a row range ``(start, stop)``, or ``None`` when nothing drained.
        """
        self.rows += 1
        drained = [h.commit() for h in self.histories.values()]
        if not any(drained):
            return None
        return (self.rows - self.staging, self.rows)
