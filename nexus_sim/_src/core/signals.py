"""Signals: the typed values that pass between the loop's components on the tick.

A stage declares each signal it reads and each signal it writes, by name and type. The builder wires
every input to the one output of its name before the capture, checks that the two agree, and hands the
writer and each reader one buffer. A device type's buffer is a Warp array, which device stages read and
write in their kernels and a host stage reads with a copy. Any other type is a host type, whose buffer
holds one object.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from .stages import Bound


class DeviceType:
    """The base of a device signal's type: its buffer is a Warp array of ``dtype`` in the signal's shape.

    A component subclasses it to declare its own device type, beside the types core ships. ``dtype`` is a
    Warp dtype, or the name of one in ``warp`` for a type core declares, since core imports no Warp.
    """

    __slots__ = ()
    dtype: ClassVar[Any]


class HostBuffer:
    """A host signal's buffer: one object, which the writer replaces and each reader reads."""

    __slots__ = ("value",)

    def __init__(self, value: Any = None):
        self.value = value


class Signal:
    """One value a stage reads or writes: a name, a type and, for a device type, a shape.

    Args:
        name: The name that wires a reader to its writer.
        type: A :class:`DeviceType` subclass, whose buffer lives on the device, or any other class, whose
            buffer holds one object on the host.
        shape: The device buffer's shape. A reader reads the leading part of a wider buffer.
        default: The value a reader takes until a component writes the signal, and for good when no
            component writes it. ``None`` gives no default.
    """

    buffer: Any
    """The one buffer the builder hands the writer and every reader, before the capture."""

    def __init__(self, name: str, type: type, *, shape: tuple[int, ...] | None = None, default: Any = None):
        self.name = name
        self.type = type
        self.shape = None if shape is None else tuple(int(n) for n in shape)
        self.default = default
        self.buffer = None

    @property
    def device(self) -> bool:
        """Whether the signal's buffer lives on the device: its type is a :class:`DeviceType`."""
        return isinstance(self.type, type) and issubclass(self.type, DeviceType)

    def read(self) -> Any:
        """The signal's value on the host: a copy of a device buffer, or the object a host buffer holds."""
        return self.buffer.numpy() if self.device else self.buffer.value

    def write(self, value: Any) -> None:
        """Write the signal's value from the host, in place: a device buffer takes a copy of ``value``, an
        array of the signal's shape, so the next graph replay reads it, and a host buffer holds ``value``.
        """
        if self.device:
            import numpy as np

            self.buffer.assign(np.asarray(value))
        else:
            self.buffer.value = value

    def _allocate(self) -> Any:
        """A new buffer from the signal's type: zeros on the current Warp device, or an empty host buffer."""
        if not self.device:
            return HostBuffer()
        import warp as wp

        dtype = self.type.dtype
        return wp.zeros(self.shape, dtype=getattr(wp, dtype) if isinstance(dtype, str) else dtype)


def wire(ring: list[Bound]) -> None:
    """Wire every signal the ring's stages declare, before any stage runs.

    Each input meets the one output of its name. The builder allocates one buffer per signal and hands it
    to the writer and to each reader, and a reader's default fills it until the writer first writes. An
    input that no component writes reads its own buffer, which holds its default.
    """
    writes: dict[str, list[Signal]] = {}
    reads: dict[str, list[Signal]] = {}
    seen: set[int] = set()
    for bound in ring:
        if id(bound) in seen:  # the ring repeats the stages of each physics substep
            continue
        seen.add(id(bound))
        for signal in bound.stage.reads:
            reads.setdefault(signal.name, []).append(signal)
        for signal in bound.stage.writes:
            writes.setdefault(signal.name, []).append(signal)
    for name, readers in reads.items():
        if name not in writes:
            for signal in readers:
                _hand(signal._allocate(), [signal], signal.default)
            continue
        written = writes[name][0]
        default = next((signal.default for signal in readers if signal.default is not None), None)
        _hand(written._allocate(), [*writes[name], *readers], default)
    for name, written in writes.items():
        if name not in reads:  # an output no component reads: each writer keeps a buffer of its own
            for signal in written:
                _hand(signal._allocate(), [signal], None)


def _hand(buffer: Any, signals: list[Signal], default: Any) -> None:
    """Hand ``buffer`` to every declaration of one signal, and write ``default`` into it, if any."""
    for signal in signals:
        signal.buffer = buffer
    if default is not None:
        signals[0].write(default)


__all__ = ["DeviceType", "HostBuffer", "Signal", "wire"]
