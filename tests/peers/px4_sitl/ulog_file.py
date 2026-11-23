"""A ULog file a test writes: its info, its parameters and its topics' rows, in PX4's file format."""

import struct
from pathlib import Path

_PACK = {
    "uint64_t": "Q",
    "uint32_t": "I",
    "uint16_t": "H",
    "uint8_t": "B",
    "int32_t": "i",
    "bool": "?",
    "float": "f",
    "double": "d",
}


def _message(kind: str, payload: bytes) -> bytes:
    return struct.pack("<HB", len(payload), ord(kind)) + payload


def _keyed(kind: str, key: str, value: bytes) -> bytes:
    return _message(kind, struct.pack("<B", len(key)) + key.encode() + value)


def write(path: Path, *, info: dict | None = None, params: dict | None = None, topics: dict | None = None) -> str:
    """Write a ULog at `path` and return its path.

    `info` maps a key to its text, such as `ver_sw` to a commit. `params` maps a parameter's name to its value,
    an `int` for an integer parameter and a `float` for a float one. `topics` maps a topic's name to its fields,
    a list of `(type, name)` with `("uint64_t", "timestamp")` first, and its rows, a list of tuples in the fields'
    order.
    """
    out = b"ULog\x01\x12\x35\x01" + struct.pack("<Q", 0) + _message("B", bytes(40))
    for key, text in (info or {}).items():
        out += _keyed("I", f"char[{len(text)}] {key}", text.encode())
    for name, value in (params or {}).items():
        is_int = isinstance(value, int)
        out += _keyed("P", f"{'int32_t' if is_int else 'float'} {name}", struct.pack("<i" if is_int else "<f", value))
    rows = []
    for msg_id, (name, (fields, topic_rows)) in enumerate((topics or {}).items()):
        out += _message("F", f"{name}:{''.join(f'{t} {f};' for t, f in fields)}".encode())
        layout = "<" + "".join(_PACK[t] for t, _ in fields)
        rows += [(row[0], msg_id, struct.pack(layout, *row)) for row in topic_rows]
    for msg_id, name in enumerate(topics or {}):
        out += _message("A", struct.pack("<BH", 0, msg_id) + name.encode())
    for _, msg_id, data in sorted(rows, key=lambda r: r[0]):
        out += _message("D", struct.pack("<H", msg_id) + data)
    Path(path).write_bytes(out)
    return str(path)
