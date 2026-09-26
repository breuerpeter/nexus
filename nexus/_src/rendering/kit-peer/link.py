"""The peer's end of the render link: length-framed JSON headers with binary blobs behind them.

A message is a 4-byte big-endian header length, the header as UTF-8 JSON, then the blobs the
header lists by size under ``blobs``. The host's end, ``nexus/_src/rendering/link.py``, frames the
same way; the two stay twins because the peer imports no nexus module.
"""

from __future__ import annotations

import json
import socket
import struct

_LEN = struct.Struct("!I")


def send(sock: socket.socket, header: dict, blobs: list = ()) -> None:
    """Send one message: ``header`` plus ``blobs``, each a bytes-like object."""
    views = [memoryview(b).cast("B") for b in blobs]
    head = json.dumps({**header, "blobs": [v.nbytes for v in views]}).encode()
    sock.sendall(_LEN.pack(len(head)) + head)
    for v in views:
        sock.sendall(v)


def _read(sock: socket.socket, n: int) -> bytearray:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("the host closed the render link")
        got += k
    return buf


def recv(sock: socket.socket) -> tuple[dict, list[bytearray]]:
    """Read one message: its header and its blobs."""
    (n,) = _LEN.unpack(_read(sock, _LEN.size))
    header = json.loads(_read(sock, n))
    return header, [_read(sock, size) for size in header.get("blobs", [])]
