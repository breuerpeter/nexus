"""The host's end of the render link, and the renderer seam the loop drives.

A message is a 4-byte big-endian header length, the header as UTF-8 JSON, then the binary blobs
the header lists by size under ``blobs``. The peer's end, ``kit-peer/link.py``, frames the same
way; the two stay twins because the peer imports no nexus module.

The link pipelines the rendering. At a frame's due tick it sends the sim time and a world matrix per
prim path, and the loop flies on; the frame comes back on a later due tick, and the loop waits only
when the peer is still rendering the frame before. A frame then costs the loop the slower of physics
and render rather than their sum, and the link drops no frame: the last one arrives at close.
"""

from __future__ import annotations

import contextlib
import json
import os
import select
import socket
import struct
import time

import numpy as np

from nexus._src.core import logger

from .peer import KitPeer, KitPeerError

_LEN = struct.Struct("!I")
STARTUP_TIMEOUT_S = 1800.0  # a cold boot compiles shaders; a Cesium scene streams its tiles first
FRAME_TIMEOUT_S = 60.0  # no frame for this long means a hung peer


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
            raise ConnectionError("the link closed")
        got += k
    return buf


def recv(sock: socket.socket) -> tuple[dict, list[bytearray]]:
    """Read one message: its header and its blobs."""
    (n,) = _LEN.unpack(_read(sock, _LEN.size))
    header = json.loads(_read(sock, n))
    return header, [_read(sock, size) for size in header.get("blobs", [])]


def pose_matrices(body_q) -> np.ndarray:
    """``body_q`` rows, (px py pz qx qy qz qw), as row-vector world matrices, shape (n, 4, 4).

    Row-vector, as Universal Scene Description (USD) ``Gf.Matrix4d`` holds a pose: the rotation is the transpose of the column-vector one,
    and the translation sits in the last row.
    """
    bq = np.asarray(body_q, dtype=np.float64).reshape(-1, 7)
    x, y, z, w = bq[:, 3], bq[:, 4], bq[:, 5], bq[:, 6]
    m = np.zeros((len(bq), 4, 4))
    m[:, 0, 0] = 1 - 2 * (y * y + z * z)
    m[:, 0, 1] = 2 * (x * y + z * w)
    m[:, 0, 2] = 2 * (x * z - y * w)
    m[:, 1, 0] = 2 * (x * y - z * w)
    m[:, 1, 1] = 1 - 2 * (x * x + z * z)
    m[:, 1, 2] = 2 * (y * z + x * w)
    m[:, 2, 0] = 2 * (x * z + y * w)
    m[:, 2, 1] = 2 * (y * z - x * w)
    m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    m[:, 3, :3] = bq[:, :3]
    m[:, 3, 3] = 1.0
    return m


class KitRenderer:
    """The Kit render peer on the loop's side: the renderer seam plus the link the RTX sensors ride.

    The loop calls :meth:`on_physics_ready` before the flight and :meth:`close` after it. Each RTX
    sensor calls :meth:`tick` from its host-seam sample; the first call at a sim time does the
    tick's work and later ones return.

    Args:
        peer: The started peer container.
        bodies: ``(model body index, stage prim path)`` for each body the render poses.
        setup: The scene and vehicle fields of the setup message; a factory can fill it later,
            before :meth:`on_physics_ready`.
        sensors: The RTX sensors, each with ``path``, ``output``, ``width``, ``height``, ``body``,
            ``mount``, ``take_due(now)`` and ``emit(arrays, t_shown)``; a factory can fill them later
            too, since each sensor holds this renderer.
    """

    def __init__(self, peer: KitPeer, *, bodies: list, setup: dict | None = None, sensors: list = ()):
        self._peer = peer
        self._bodies = list(bodies)
        self.setup = dict(setup or {})
        self.sensors = list(sensors)
        self._sock: socket.socket | None = None
        self._pending = False  # a frame in flight: this link asked for it and hasn't taken it yet
        self._last_t: float | None = None
        self._prof = None

    def set_profiler(self, prof) -> None:
        """The loop's profiler, for the send and wait spans."""
        self._prof = prof

    def _span(self, name: str):
        return self._prof.span(name) if self._prof is not None else contextlib.nullcontext()

    def _died(self, what: str) -> str:
        tail = self._peer.console_tail()
        return f"the Kit render peer {what}; its console is {self._peer.log_path}" + (f":\n{tail}" if tail else "")

    def _connect(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while True:
            if not self._peer.alive():
                raise KitPeerError(self._died("exited while starting"))
            sock = None
            try:
                sock = socket.create_connection(("127.0.0.1", self._peer.port), timeout=5.0)
                header, _ = recv(sock)  # the peer greets the moment it accepts
                if header.get("op") == "hello":
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self._sock = sock
                    return
            except (OSError, ConnectionError, ValueError):
                pass  # not listening yet: docker's proxy accepts and closes until it does
            if sock is not None:
                sock.close()
            if time.monotonic() > deadline:
                raise KitPeerError(self._died(f"did not answer within {STARTUP_TIMEOUT_S:.0f}s"))
            time.sleep(0.5)

    def _receive(self, timeout: float, what: str) -> tuple[dict, list]:
        """The peer's next message, waiting at most ``timeout`` and failing at once if the peer dies."""
        deadline = time.monotonic() + timeout
        while not select.select([self._sock], [], [], 1.0)[0]:
            if not self._peer.alive():
                raise ConnectionError(self._died(f"died while this run waited for {what}"))
            if time.monotonic() > deadline:
                raise ConnectionError(self._died(f"sent no {what} within {timeout:.0f}s"))
        self._sock.settimeout(FRAME_TIMEOUT_S)
        try:
            header, blobs = recv(self._sock)
        except (OSError, ConnectionError) as exc:
            raise ConnectionError(self._died(f"closed the link while this run waited for {what} ({exc})")) from exc
        if header.get("op") == "error":
            raise ConnectionError(self._died(f"failed: {header.get('message')}"))
        return header, blobs

    def on_physics_ready(self) -> None:
        """Connect to the peer, send the setup, and wait until it has composed and warmed the stage.

        Raises:
            KitPeerError: The peer exited, failed its setup, or never answered; the message names
                the cause and carries the tail of its console.
        """
        t0 = time.monotonic()
        self._connect()
        sensors = [{"path": s.path, "output": s.output, "width": s.width, "height": s.height} for s in self.sensors]
        # The Cesium ion token crosses here and nowhere else: never in the container's environment.
        token = os.environ.get("CESIUM_ION_TOKEN") or None
        send(self._sock, {"op": "setup", **self.setup, "cesium_ion_token": token, "sensors": sensors})
        try:
            header, _ = self._receive(STARTUP_TIMEOUT_S, "the ready")
        except ConnectionError as exc:
            raise KitPeerError(str(exc)) from exc
        if header.get("op") != "ready":
            raise KitPeerError(self._died(f"answered the setup with {header.get('op')!r}"))
        logger.info(f"Kit render peer ready in {time.monotonic() - t0:.0f}s: {len(self.sensors)} RTX sensor(s)")

    def _take(self) -> None:
        """Take the frame in flight, if any, and hand each output to its sensor."""
        if not self._pending:
            return
        self._pending = False
        with self._span("render.wait"):
            header, blobs = self._receive(FRAME_TIMEOUT_S, "a frame")
        outputs: dict[int, dict] = {}
        for spec, blob in zip(header["arrays"], blobs, strict=True):
            arr = np.frombuffer(blob, dtype=spec["dtype"]).reshape(spec["shape"])
            outputs.setdefault(int(spec["sensor"]), {})[spec["name"]] = arr
        for i, arrays in outputs.items():
            self.sensors[i].emit(arrays, float(header["t"]))

    def tick(self, t, state) -> None:
        """At a tick where any sensor is due: take the frame before, then send this one's poses.

        Raises:
            ConnectionError: The peer died or failed; the message names it, and the loop ends the
                run as it does for a lost autopilot.
        """
        now = float(t.sim_time)
        if now == self._last_t:
            return  # a sibling sensor already did this tick's work
        self._last_t = now
        due = [i for i, s in enumerate(self.sensors) if s.take_due(now)]
        if not due:
            return
        self._take()
        with self._span("render.send"):
            worlds = pose_matrices(state.body_q.numpy())  # the host copy the render needs: a few bodies
            mats = [worlds[b] for b, _ in self._bodies] + [s.mount @ worlds[s.body] for s in self.sensors]
            paths = [p for _, p in self._bodies] + [s.path for s in self.sensors]
            try:
                send(self._sock, {"op": "frame", "t": now, "paths": paths, "due": due}, [np.stack(mats)])
            except OSError as exc:
                raise ConnectionError(self._died(f"closed the link ({exc})")) from exc
        self._pending = True

    def close(self) -> None:
        """Take the last frame, end the peer, and remove its container. Never raises: it runs in teardown."""
        try:
            if self._sock is not None:
                self._take()
                send(self._sock, {"op": "close"})
                self._receive(30.0, "the close")
        except Exception as exc:
            logger.warning(f"Kit render peer teardown: {exc}")
        finally:
            if self._sock is not None:
                self._sock.close()
                self._sock = None
            self._peer.stop()
