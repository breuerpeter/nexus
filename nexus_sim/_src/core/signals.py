"""Signals: the typed values that pass between the loop's components on the tick.

A stage declares each signal it reads and each signal it writes, by name and type. The builder wires
every input to the one output of its name before the capture, checks that the two agree, and hands the
writer and each reader one buffer. A device type's buffer is a Warp array, which device stages read and
write in their kernels and a host stage reads with a copy. Any other type is a host type, whose buffer
holds one object.
"""

from __future__ import annotations

from typing import Any, ClassVar


class DeviceType:
    """The base of a device signal's type: its buffer is a Warp array of ``dtype`` in the signal's shape.

    A component subclasses it to declare its own device type, beside the types core ships.
    """

    dtype: ClassVar[Any]


class Signal:
    """One value a stage reads or writes: a name, a type and, for a device type, a shape.

    Args:
        name: The name that wires a reader to its writer.
        type: A :class:`DeviceType` subclass, whose buffer lives on the device, or any other class, whose
            buffer holds one object on the host.
        shape: The device buffer's shape. A reader reads the leading part of a wider buffer.
        default: The value a reader takes when no component writes the signal.
    """

    buffer: Any
    """The one buffer the builder hands the writer and every reader, before the capture."""

    def __init__(self, name: str, type: type, *, shape: tuple[int, ...] | None = None, default: Any = None): ...
