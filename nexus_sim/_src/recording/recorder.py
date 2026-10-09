"""Recorder and History: every row a run records, with a fixed footprint on the device.

A :class:`History` is one body's, joint's or signal's rows over a run. A device kernel writes one row per
tick into its staging buffers, the row's values and its sim time, at the slot the device counter names, so
the write joins the captured graph with no host readback. The loop commits one row per tick on the host,
and each time the staging buffers fill they drain: the rows copy onto the end of the host blocks, which
grow with the run, and the kernel writes slot 0 again. So device memory stays at ``staging`` rows per
history and the Recorder drops no row. The host reads a history on demand: ``latest`` reads the newest row
off the device, ``history`` every row, the blocks then the rows still staged.

The :class:`Recorder` holds the histories by key and records everything itself: :meth:`watch` registers
one history per body and per joint from the physics' model, :meth:`tap` one per device signal a component
writes, under the writer's path then the signal's name, :meth:`record` snapshots them all in the record
stage, and :meth:`commit` counts the tick's row and drains the staging buffers when full. No component
states a recording member.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import warp as wp

from nexus_sim._src.core.labels import leaf_keys
from nexus_sim._src.core.signals import Signal

from .state import BODY_FIELDS, BODY_WIDTH, decode_body, make_decode_joint, record_body, record_joint

STAGING = 4096  # rows a staging buffer holds: about 16 s at 250 Hz, the most a SIGKILL can lose


@wp.kernel
def record_signal(
    src: wp.array(dtype=Any),
    dt: wp.float64,
    staging: int,
    values: wp.array2d(dtype=Any),
    times: wp.array(dtype=wp.float64),
    counter: wp.array(dtype=int),
):
    """Copy a signal's buffer, flat, into the row the device counter names, and stamp the row's time. One
    generic kernel serves every signal type: a struct, a value type or a row of floats.
    """
    c = counter[0]  # rows written so far; advances per graph replay, on the device, not a frozen kernel arg
    s = c % staging  # the staging slot: the host drains the buffer before the slot comes round again
    times[s] = dt * wp.float64(c)
    for k in range(src.shape[0]):
        values[s, k] = src[k]
    counter[0] = c + 1


class History:
    """One body's, joint's or signal's rows over a run.

    A row is ``width`` values of ``dtype``, a Warp element type, and its sim time. The writing kernel takes
    :attr:`values`, the device staging buffer of ``staging`` rows, :attr:`times`, their sim times,
    :attr:`counter`, the rows written so far, which advances on the device, and ``dt``, and stamps
    ``times[slot] = dt × counter``: a kernel argument freezes at capture and the counter doesn't. The host
    reconstructs nothing.

    ``fields`` is the declared row schema: ordered ``(name, n_values)`` pairs that cover the row, such as
    ``(("position", 3), ("quat_xyzw", 4), …)`` for a body or a struct's fields for a signal. It's the one
    declaration everything downstream derives from: :meth:`arrays`, the per-quantity readback, and the
    series the Logger writes. ``decode`` turns one row, its time and its values, into the read type the
    plant keeps, ``BodyState`` or ``JointState``; with none, the rows read as records of ``t`` then the
    fields, a signal's shape. ``source`` names the writing class, ``ImuSensor`` say: kind metadata for tools.
    ``path`` is the writer's path below the process root and ``row`` the row's name under it, the signal's,
    or ``None`` for the plant, whose key is its path.
    """

    def __init__(
        self,
        width: int,
        dt: float,
        decode: Callable[[float, np.ndarray], object] | None,
        staging: int = STAGING,
        dtype: Any = float,
        fields: tuple[tuple[str, int], ...] | None = None,
        source: str | None = None,
        path: str | None = None,
        row: str | None = None,
    ):
        self.width = int(width)
        self.dt = float(dt)
        self.staging = int(staging)
        self.dtype = dtype
        self.fields = tuple((str(n), int(w)) for n, w in fields) if fields is not None else None
        self.source = source
        self.path = path
        self.row = row
        self.values = wp.zeros((self.staging, self.width), dtype=dtype)  # the staged rows' values
        self.times = wp.zeros(self.staging, dtype=wp.float64)  # the staged rows' sim times
        self.counter = wp.zeros(1, dtype=int)  # rows written so far; advances IN the graph
        self.rows = 0  # rows the loop has committed: the host's count, which decides the drains
        self._decode = decode
        self._blocks: list[tuple[np.ndarray, np.ndarray]] = []  # the drained (times, values), oldest first
        self._lock = threading.Lock()  # serialise the readback: one consistent (counter, buffers) per reader
        if self.fields is not None and not self._structured and sum(w for _, w in self.fields) != self._columns:
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

    def latest(self) -> object:
        """Return the most recently written row, decoded, or as a record for a signal's history.

        Reads one row off the device, not the whole staging buffer. That matters because this is the
        per-tick read: ``Sim.wait_until`` calls it every control tick for the sim clock.

        Raises:
            RuntimeError: No row exists yet: sim not started or not stepped.
        """
        with self._lock:
            c = int(self.counter.numpy()[0])
            if c == 0:
                raise RuntimeError("History has no row yet (sim not started / not stepped)")
            slot = (c - 1) % self.staging
            t, row = self.times[slot : slot + 1].numpy(), self.values[slot : slot + 1].numpy()
        return self._decoded(t, row)[0]

    def history(self) -> object:
        """Return every row, oldest first: decoded, or as one structured array for a signal's history."""
        return self._decoded(*self._rows(0, None))

    def _decoded(self, times: np.ndarray, values: np.ndarray) -> object:
        if self._decode is not None:
            return [self._decode(float(t), v) for t, v in zip(times, values, strict=True)]
        return self._records(times, values)

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

        Raises:
            ValueError: The history declared no ``fields`` schema.
        """
        if self.fields is None:
            raise ValueError("history declared no fields schema (register with fields=...)")
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
    then its row's name for a signal: ``vehicle/body/<label>`` and ``vehicle/joints/<label>`` for the
    plant, which :meth:`watch` registers, and ``<path>/<signal>`` for a signal, such as
    ``vehicle/sensors/imu/imu``, which :meth:`tap` registers. ``base_body`` is the base body's history,
    the flown path's source.
    """

    def __init__(self, dt: float, staging: int = STAGING):
        self._dt = float(dt)
        self.staging = int(staging)
        self.histories: dict[str, History] = {}
        self.base_body: History | None = None
        self.rows = 0  # rows committed so far, the same for every history
        self._body_taps: list[tuple[History, int]] = []
        self._joint_taps: list[tuple[History, int, int, int, int]] = []
        self._signal_taps: list[tuple[History, wp.array]] = []
        self._state = None

    def history(
        self,
        name: str,
        *,
        width: int,
        decode: Callable[[float, np.ndarray], object],
        dtype: Any = float,
        fields: tuple[tuple[str, int], ...] | None = None,
        source: str | None = None,
    ) -> History:
        """Register a history of float rows, the plant's: a ``width``-element row buffer with its decode and
        its declared ``fields`` schema and ``source`` metadata; see :class:`History`.

        Idempotent for a re-registration with the same width/dtype/fields/source, which returns the
        existing history, but a duplicate name with a different shape raises: two components must never
        silently share one buffer, for example two Inertial Measurement Unit (IMU) instances both named
        ``imu``; give them distinct instance names instead: ``imu_bosch`` / ``imu_murata``.
        """
        h = self.histories.get(name)
        if h is not None:
            declared = tuple((str(n), int(w)) for n, w in fields) if fields is not None else None
            if (h.width, h.dtype, h.fields, h.source) != (int(width), dtype, declared, source):
                raise ValueError(
                    f"history {name!r} is already registered with a different shape "
                    f"(width={h.width}, dtype={h.dtype}, fields={h.fields}, source={h.source}): "
                    "distinct components need distinct instance names"
                )
            return h
        h = History(width, self._dt, decode, self.staging, dtype, fields=fields, source=source, path=name)
        self.histories[name] = h
        return h

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
            self._dt,
            None,
            self.staging,
            signal.type,
            fields=_quantities(signal, width),
            source=source,
            path=path,
            row=signal.name,
        )
        self.histories[key] = h
        self._signal_taps.append((h, src))
        return h

    def watch(self, physics) -> None:
        """Register one history per body, ``vehicle/body/<label>``, and per joint, ``vehicle/joints/<label>``,
        of the plant's model, from its labels: every body and joint addressable by name. The base body's
        history, that of ``physics.base_index``, becomes :attr:`base_body`. A physics with no model, a
        stand-in, registers nothing.
        """
        model = getattr(physics, "model", None)
        if model is None:
            return
        self._state = physics.current_state
        src = type(physics).__name__
        body_keys = leaf_keys(
            [str(k) for k in model.body_label]
        )  # friendly names: leaf when unique, else the full path
        self._body_taps = [
            (
                self.history(
                    f"vehicle/body/{key}", width=BODY_WIDTH, decode=decode_body, fields=BODY_FIELDS, source=src
                ),
                i,
            )
            for i, key in enumerate(body_keys)
        ]
        self.base_body = self._body_taps[int(physics.base_index)][0]
        self._joint_taps = []
        if model.joint_count and self._state.joint_q is not None:
            joint_keys = leaf_keys([str(k) for k in model.joint_label])
            qs = model.joint_q_start.numpy()  # length joint_count+1, with a sentinel, → clean per-joint slices
            qds = model.joint_qd_start.numpy()
            for j, key in enumerate(joint_keys):
                nq, nqd = int(qs[j + 1] - qs[j]), int(qds[j + 1] - qds[j])
                h = self.history(
                    f"vehicle/joints/{key}",
                    width=nq + nqd,
                    decode=make_decode_joint(nq, nqd),
                    fields=(("q", nq), ("qd", nqd)),
                    source=src,
                )
                self._joint_taps.append((h, int(qs[j]), nq, int(qds[j]), nqd))

    def record(self) -> None:
        """The record taps: snapshot every body and joint of the live state, and every tapped signal, into
        their histories, device-only, so the launches join the captured graph. The loop's record stage runs
        this each tick.
        """
        if self._state is not None:
            bq, bqd = self._state.body_q, self._state.body_qd
            for h, i in self._body_taps:
                wp.launch(record_body, dim=1, inputs=(bq, bqd, i, h.dt, h.staging, h.values, h.times, h.counter))
            if self._joint_taps:
                jq, jqd = self._state.joint_q, self._state.joint_qd
                for h, qs, nq, qds, nqd in self._joint_taps:
                    wp.launch(
                        record_joint,
                        dim=1,
                        inputs=(jq, jqd, qs, nq, qds, nqd, h.dt, h.staging, h.values, h.times, h.counter),
                    )
        for h, src in self._signal_taps:  # the generic kernel infers a Python float as float32, so cast
            wp.launch(record_signal, dim=1, inputs=(src, wp.float64(h.dt), h.staging, h.values, h.times, h.counter))

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


class Histories(Mapping):
    """A read-only, name-keyed view over a subset of a Recorder's histories: the host-facing surface.

    ``Sim.physics`` and ``Sim.sensors`` are views over the flat ``Recorder.histories`` dict, selected by
    the prefix of a history's path and addressed by the bare instance name. ``sim.physics["rotor_1_joint"]``
    resolves ``histories["vehicle/joints/rotor_1_joint"]``, and ``Sim.physics`` spans both ``vehicle/body/``
    and ``vehicle/joints/``. A sensor's instance resolves the one signal's history under its path:
    ``sim.sensors["imu"]`` resolves ``histories["vehicle/sensors/imu/imu"]``. A lookup returns the
    :class:`History`, so the caller picks ``.latest()`` or ``.history()``.
    """

    def __init__(self, histories: dict[str, History], prefixes: tuple[str, ...]):
        self._histories = histories
        self._prefixes = prefixes

    def _resolve(self, name: str) -> str:
        paths = {p + name for p in self._prefixes}
        hits = [key for key, h in self._histories.items() if h.path in paths]
        if not hits:
            raise KeyError(name)
        if len(hits) > 1:  # the same bare name under two namespaces, say a body and a joint: ambiguous
            raise KeyError(f"ambiguous entity name {name!r}: matches {hits}")
        return hits[0]

    def __getitem__(self, name: str) -> History:
        return self._histories[self._resolve(name)]

    def __iter__(self):
        seen = set()
        for h in self._histories.values():
            for p in self._prefixes:
                if h.path.startswith(p) and h.path not in seen:
                    seen.add(h.path)
                    yield h.path[len(p) :]

    def __len__(self) -> int:
        return sum(1 for _ in self)
