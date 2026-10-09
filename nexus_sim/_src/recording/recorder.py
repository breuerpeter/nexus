"""Recorder and History: every row a run records, with a fixed footprint on the device.

A :class:`History` is one body's, joint's or sensor's rows over a run. A device kernel writes one
``width``-float row per tick into its staging buffer, at the slot the device counter names, so the
write joins the captured graph with no host readback. The loop commits one row per tick on the host,
and each time the staging buffer fills it drains: the rows copy onto the end of the host blocks, which
grow with the run, and the kernel writes slot 0 again. So device memory stays at ``staging`` rows per
history and the Recorder drops no row. The host reads a history on demand: ``latest`` reads the newest row
off the device, ``history`` every row, the blocks then the rows still staged.

The :class:`Recorder` holds the histories by name and records the plant itself: :meth:`watch` registers
one history per body and per joint from the physics' model, :meth:`record` snapshots them in the
record stage, and :meth:`commit` counts the tick's row and drains the staging buffers when full. A sensor still registers its own history through :meth:`history` and writes it with its own
kernel, until #250 moves that here too.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping

import numpy as np
import warp as wp

from nexus_sim._src.core.labels import leaf_keys

from .state import BODY_FIELDS, BODY_WIDTH, decode_body, make_decode_joint, record_body, record_joint

STAGING = 4096  # rows a staging buffer holds: about 16 s at 250 Hz, the most a SIGKILL can lose


class History:
    """One body's, joint's or sensor's rows over a run.

    The writing kernel takes :attr:`buf`, the device staging buffer of ``staging`` rows, :attr:`counter`,
    the rows written so far, which advances on the device, and ``dt``, and stamps ``row[0] = dt ×
    counter``, the row's sim time: a kernel argument freezes at capture and the counter doesn't. The
    host reconstructs nothing. ``dtype`` is the row element type, ``float`` for f32 by default; a Global
    Positioning System (GPS) history takes ``wp.float64`` so lat/lon survive, and its kernel's buffer
    argument matches.

    ``fields`` is the declared row schema: ordered ``(name, n_columns)`` pairs covering the columns after
    the implicit time column 0, for example ``(("position", 3), ("quat_xyzw", 4), …)``. It's the one
    declaration everything downstream derives from: :meth:`arrays`, the per-quantity readback, and the
    series the Logger writes. ``source`` names the registering class, ``ImuSensor`` say: kind metadata
    for tools; the history's key ends in the flat instance name, so two instances of one class are
    siblings, as in ``vehicle/sensors/imu_bosch``.
    """

    def __init__(
        self,
        width: int,
        dt: float,
        decode: Callable[[np.ndarray], object],
        staging: int = STAGING,
        dtype: type = float,
        fields: tuple[tuple[str, int], ...] | None = None,
        source: str | None = None,
    ):
        self.width = int(width)
        self.dt = float(dt)
        self.staging = int(staging)
        self.dtype = dtype
        self.fields = tuple((str(n), int(w)) for n, w in fields) if fields is not None else None
        if self.fields is not None and 1 + sum(w for _, w in self.fields) != self.width:
            raise ValueError(
                f"fields {self.fields} cover {1 + sum(w for _, w in self.fields)} columns "
                f"(incl. the implicit t) but width is {self.width}"
            )
        self.source = source
        self.buf = wp.zeros((self.staging, width), dtype=dtype)  # the device staging buffer
        self.counter = wp.zeros(1, dtype=int)  # rows written so far; advances IN the graph
        self.rows = 0  # rows the loop has committed: the host's count, which decides the drains
        self._decode = decode
        self._blocks: list[np.ndarray] = []  # the drained rows, oldest first, ``staging`` rows a block
        self._lock = threading.Lock()  # serialise the readback: one consistent (counter, buf) pair per reader

    @property
    def _drained(self) -> int:
        return len(self._blocks) * self.staging

    def commit(self) -> bool:
        """Count the row the record stage wrote this tick, and drain the staging buffer when full.

        Returns:
            Whether a drain happened: the last ``staging`` rows are now a host block.
        """
        self.rows += 1
        if self.rows - self._drained < self.staging:
            return False
        with self._lock:
            self._blocks.append(self.buf.numpy().copy())  # one device-to-host copy per ``staging`` ticks
        return True

    def _snapshot(self) -> tuple[int, np.ndarray]:
        with self._lock:  # one consistent readback; .numpy() syncs, so it's the last completed tick's rows
            return int(self.counter.numpy()[0]), self.buf.numpy()

    def latest(self) -> object:
        """Return the most recently written row, decoded.

        Reads one row off the device, not the whole staging buffer. That matters because this is the
        per-tick read: ``Sim.wait_until`` calls it every control tick for the sim clock.

        Raises:
            RuntimeError: No row exists yet: sim not started or not stepped.
        """
        with self._lock:
            c = int(self.counter.numpy()[0])
            if c == 0:
                raise RuntimeError("History has no row yet (sim not started / not stepped)")
            row = self.buf[(c - 1) % self.staging].numpy()
        return self._decode(row)

    def history(self) -> list:
        """Return every row, oldest first, each decoded."""
        return [self._decode(r) for r in self._rows(0, None)]

    def _rows(self, start: int, stop: int | None) -> np.ndarray:
        """The raw rows ``[start, stop)``, ``(N, width)``, oldest first: the host blocks that overlap the
        range, then the rows still on the device, read with one locked device-to-host copy.
        """
        c, buf = self._snapshot()
        drained = self._drained
        stop = c if stop is None else min(int(stop), c)
        parts = []
        for i, block in enumerate(self._blocks):
            lo, hi = i * self.staging, (i + 1) * self.staging
            if hi > start and lo < stop:
                parts.append(block[max(start, lo) - lo : min(stop, hi) - lo])
        if stop > drained:
            parts.append(buf[max(start, drained) - drained : stop - drained])
        if not parts:
            return np.empty((0, self.width), dtype=buf.dtype)
        return np.concatenate(parts)

    def arrays(self, start: int = 0, stop: int | None = None) -> dict[str, np.ndarray]:
        """The rows ``[start, stop)`` as one array per declared quantity, ``{"t": (N,), name: (N,) |
        (N, w)}``, oldest first, straight off the rows with no per-row decode and the history's dtype.
        Width-1 quantities come back 1-D. ``N == 0`` before the first tick.

        Raises:
            ValueError: The history declared no ``fields`` schema.
        """
        if self.fields is None:
            raise ValueError("history declared no fields schema (register with fields=...)")
        rows = self._rows(start, stop)
        out = {"t": rows[:, 0].copy()}
        col = 1
        for name, w in self.fields:
            out[name] = rows[:, col].copy() if w == 1 else rows[:, col : col + w].copy()
            col += w
        return out

    def history_arrays(self) -> dict[str, np.ndarray]:
        """Every row as one array per declared quantity, as :meth:`arrays` over the whole run."""
        return self.arrays()


class Recorder:
    """The histories of a run, by name, and the plant's record taps.

    A history's name is its instance's path below the process root, the path its series take in a
    recording: ``vehicle/body/<label>`` and ``vehicle/joints/<label>`` for the plant, which :meth:`watch`
    registers, and ``vehicle/sensors/<name>`` for a sensor, which registers its own through
    :meth:`history`. ``base_body`` is the base body's history, the flown path's source.
    """

    def __init__(self, dt: float, staging: int = STAGING):
        self._dt = float(dt)
        self.staging = int(staging)
        self.histories: dict[str, History] = {}
        self.base_body: History | None = None
        self.rows = 0  # rows committed so far, the same for every history
        self._body_taps: list[tuple[History, int]] = []
        self._joint_taps: list[tuple[History, int, int, int, int]] = []
        self._state = None

    def history(
        self,
        name: str,
        *,
        width: int,
        decode: Callable[[np.ndarray], object],
        dtype: type = float,
        fields: tuple[tuple[str, int], ...] | None = None,
        source: str | None = None,
    ) -> History:
        """Register a history: a ``width``-element row buffer with its decode and its declared
        ``fields`` schema and ``source`` metadata; see :class:`History`.

        Idempotent for a re-registration with the same width/dtype/fields/source, which returns the
        existing history, but a duplicate name with a different shape raises: two components must never
        silently share one buffer, for example two Inertial Measurement Unit (IMU) instances both named
        ``imu``; give them distinct instance names instead: ``imu_bosch`` / ``imu_murata``.

        ``dtype`` is the row element type: default ``float`` = f32, and a GPS history passes ``wp.float64``.
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
        h = History(width, self._dt, decode, self.staging, dtype, fields=fields, source=source)
        self.histories[name] = h
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
                    width=1 + nq + nqd,
                    decode=make_decode_joint(nq, nqd),
                    fields=(("q", nq), ("qd", nqd)),
                    source=src,
                )
                self._joint_taps.append((h, int(qs[j]), nq, int(qds[j]), nqd))

    def record(self) -> None:
        """The plant's record taps: snapshot every body and joint of the live state into their histories,
        device-only, so the launches join the captured graph. The loop's record stage runs this each tick.
        """
        if self._state is None:
            return
        bq, bqd = self._state.body_q, self._state.body_qd
        for h, i in self._body_taps:
            wp.launch(record_body, dim=1, inputs=(bq, bqd, i, h.dt, h.staging, h.buf, h.counter))
        if self._joint_taps:
            jq, jqd = self._state.joint_q, self._state.joint_qd
            for h, qs, nq, qds, nqd in self._joint_taps:
                wp.launch(record_joint, dim=1, inputs=(jq, jqd, qs, nq, qds, nqd, h.dt, h.staging, h.buf, h.counter))

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
    key prefix and addressed by the bare entity name. A history's key is its instance's path below the
    process root, the path its series take in a recording: ``sim.physics["rotor_1_joint"]`` resolves
    ``histories["vehicle/joints/rotor_1_joint"]``, and ``Sim.physics`` spans both ``vehicle/body/`` and
    ``vehicle/joints/``. A lookup returns the :class:`History`, so the caller picks ``.latest()`` or
    ``.history()``.
    """

    def __init__(self, histories: dict[str, History], prefixes: tuple[str, ...]):
        self._histories = histories
        self._prefixes = prefixes

    def _resolve(self, name: str) -> str:
        hits = [p + name for p in self._prefixes if (p + name) in self._histories]
        if not hits:
            raise KeyError(name)
        if len(hits) > 1:  # the same bare name under two namespaces, say a body and a joint: ambiguous
            raise KeyError(f"ambiguous entity name {name!r}: matches {hits}")
        return hits[0]

    def __getitem__(self, name: str) -> History:
        return self._histories[self._resolve(name)]

    def __iter__(self):
        for key in self._histories:
            for p in self._prefixes:
                if key.startswith(p):
                    yield key[len(p) :]

    def __len__(self) -> int:
        return sum(1 for _ in self)
