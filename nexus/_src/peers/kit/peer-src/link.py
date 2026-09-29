"""The render link's one definition: its framing and its message set.

A message is a 4-byte big-endian header length, the header as UTF-8 JSON, then the blobs the
header lists by size under ``blobs``. The header names its message under ``op``, one of the seven
below. The peer program imports this file as a sibling module, and the host's end,
``nexus/_src/rendering/link.py``, runs it by path, since the peer imports no nexus module.
"""

from __future__ import annotations

import json
import socket
import struct

_LEN = struct.Struct("!I")

HELLO = "hello"  # peer to host, on accept: the peer is up
SETUP = "setup"  # host to peer: the scene, the vehicle, its start pose and the sensors
READY = "ready"  # peer to host: the peer has composed the stage and warmed the renderer
FRAME = "frame"  # host to peer: the sim time and the poses; peer to host: the outputs of the frame before
CLOSE = "close"  # host to peer: render the last request, answer with its frame, and end
CLOSED = "closed"  # peer to host, the answer to a close: the outputs of the last request
ERROR = "error"  # peer to host: the setup or a render failed; ``message`` names why


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
