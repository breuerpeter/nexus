"""The fake Kit render peer: a stand-in that speaks the render link and starts no container.

It serves the link from a thread in this process, through the link's one definition,
``peer-src/link.py``, as the real peer does. It answers each ``frame`` request one frame behind,
as Kit renders, with a synthetic output at the size each sensor declares. So the renderer, the RTX
sensors and the loop run as they do against Kit, and a test proves their side of the link with no
image, no GPU and no container. It proves nothing about Kit's rendering.

A test sends the Kit peer to this class through the builder's peer mapping.
"""

from __future__ import annotations

import runpy
import socket
import threading

import numpy as np

from .runner import KIT_DIR

_wire = runpy.run_path(str(KIT_DIR / "link.py"))


def _outputs(sensor: dict) -> list[tuple[str, np.ndarray]]:
    """A synthetic frame for one sensor: each array the real peer returns for its output, blank."""
    h, w = int(sensor["height"]), int(sensor["width"])
    if sensor["output"] == "color":
        return [("color", np.zeros((h, w, 3), dtype=np.uint8))]
    if sensor["output"] == "radiance_depth":
        return [("radiance", np.zeros((h, w), dtype=np.float32)), ("depth", np.ones((h, w), dtype=np.float32))]
    return [("points", np.zeros((0, 3), dtype=np.float32))]


class KitFake:
    """A stand-in for one run's Kit render peer, on the peer contract.

    Args:
        files: The scene description files the run renders; the fake reads none of them.
        cache_dir: The run's asset cache; the fake reads nothing from it.
        error_after: Answer the frame request after this many frames with ``error``, as a peer whose
            render failed; ``None`` never does.
        error: The message the ``error`` carries.
    """

    def __init__(
        self,
        files: list = (),
        *,
        cache_dir=None,
        error_after: int | None = None,
        error: str = "the fake Kit peer failed",
    ) -> None:
        self.error_after = error_after
        self.error = error
        self.port: int | None = None
        self.log_path = None  # no console: the fake runs in this process
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stopped = False

    def start(self) -> None:
        """Listen on a free local port and serve the link in the background."""
        self._listener = socket.create_server(("127.0.0.1", 0))
        self.port = self._listener.getsockname()[1]
        self._stopped = False
        self._thread = threading.Thread(target=self._serve, name="kit-fake", daemon=True)
        self._thread.start()

    def alive(self) -> bool:
        """Whether the fake is up: from its start until its stop."""
        return self._thread is not None and not self._stopped

    def stop(self) -> None:
        """Close the link and end the fake. Idempotent."""
        self._stopped = True
        if self._listener is not None:
            self._listener.close()
            self._listener = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def console_tail(self, lines: int = 20) -> str:
        """The fake has no console."""
        return ""

    def _reply(self, conn, op: str, t, due: list, sensors: list) -> None:
        outputs = [(i, name, arr) for i in due for name, arr in _outputs(sensors[i])]
        arrays = [
            {"sensor": i, "name": name, "dtype": str(arr.dtype), "shape": list(arr.shape)} for i, name, arr in outputs
        ]
        _wire["send"](conn, {"op": op, "t": t, "arrays": arrays}, [arr for _, _, arr in outputs])

    def _serve(self) -> None:
        try:
            conn, _ = self._listener.accept()
        except OSError:
            return  # stopped before the renderer connected
        with conn:
            try:
                _wire["send"](conn, {"op": _wire["HELLO"]})
                setup, _ = _wire["recv"](conn)
                sensors = setup.get("sensors", [])
                _wire["send"](conn, {"op": _wire["READY"]})
                shown = None  # (t, due) of the request before: what the next reply renders
                frames = 0
                while not self._stopped:
                    header, _ = _wire["recv"](conn)
                    if header["op"] == _wire["CLOSE"]:
                        t, due = shown if shown else (None, [])
                        self._reply(conn, _wire["CLOSED"], t, due, sensors)
                        return
                    if self.error_after is not None and frames >= self.error_after:
                        _wire["send"](conn, {"op": _wire["ERROR"], "message": self.error})
                        return
                    t, due = shown if shown else (header["t"], [])
                    self._reply(conn, _wire["FRAME"], t, due, sensors)
                    shown = (header["t"], header["due"])
                    frames += 1
            except (OSError, ConnectionError):
                return  # the renderer closed the link
