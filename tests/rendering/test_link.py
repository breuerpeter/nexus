"""The render link: pipelined frames between the loop and the Kit peer.

A stand-in peer speaks the wire through the Kit program's own framing, ``kit-peer/link.py``, on a
real socket, so these tests also hold the two ends of the wire to one format.
"""

import runpy
import socket
import time
from types import SimpleNamespace

import numpy as np
import pytest

from nexus._src.rendering.link import KitRenderer
from nexus._src.rendering.peer import KIT_DIR, KitPeer

wire = runpy.run_path(str(KIT_DIR / "link.py"))


def _peer_serving(delay_s=0.0, die_after_setup=False):
    """A stand-in Kit peer: answers every frame with a 2x2 color image stamped with the request's time."""

    def serve(port):
        with socket.create_server(("127.0.0.1", port)) as listener:
            conn, _ = listener.accept()
            with conn:
                wire["send"](conn, {"op": "hello"})
                wire["recv"](conn)  # setup
                wire["send"](conn, {"op": "ready"})
                if die_after_setup:
                    return
                while True:
                    try:
                        header, _ = wire["recv"](conn)
                    except ConnectionError:
                        return
                    if header["op"] == "close":
                        wire["send"](conn, {"op": "closed"})
                        return
                    time.sleep(delay_s)
                    image = np.zeros((2, 2, 3), dtype=np.uint8)
                    arrays = [
                        {"sensor": i, "name": "color", "dtype": "uint8", "shape": [2, 2, 3]} for i in header["due"]
                    ]
                    wire["send"](conn, {"op": "frame", "t": header["t"], "arrays": arrays}, [image] * len(arrays))

    return serve


class _Camera:
    """A sensor riding the link: due at every tick, it keeps the stamp of every frame it gets."""

    path = "/Vehicle/body/Cam"
    output = "color"
    width = height = 2
    body = 0
    mount = np.eye(4)

    def __init__(self):
        self.frames = []

    def take_due(self, now):
        return True

    def emit(self, arrays, t_shown):
        self.frames.append(t_shown)


def _state():
    return SimpleNamespace(body_q=SimpleNamespace(numpy=lambda: np.array([[0, 0, 1, 0, 0, 0, 1.0]])))


def _tick(t):
    return SimpleNamespace(sim_time=t)


def _renderer(daemon, tmp_path, serve):
    daemon.serve = serve
    usd = tmp_path / "v.usda"
    usd.write_text("#usda 1.0\n")
    peer = KitPeer([usd], cache_dir=tmp_path / "cache")
    peer.start()
    camera = _Camera()
    renderer = KitRenderer(peer, setup={}, bodies=[(0, "/Vehicle/body")], sensors=[camera])
    renderer.on_physics_ready()
    return renderer, camera


def test_every_requested_frame_reaches_its_sensor_the_last_at_close(daemon, tmp_path):
    renderer, camera = _renderer(daemon, tmp_path, _peer_serving())
    for t in (0.1, 0.2, 0.3):
        renderer.tick(_tick(t), _state())
    renderer.close()
    assert camera.frames == [0.1, 0.2, 0.3]


def test_a_due_tick_returns_before_the_peer_answers(daemon, tmp_path):
    renderer, _ = _renderer(daemon, tmp_path, _peer_serving(delay_s=1.0))
    t0 = time.monotonic()
    renderer.tick(_tick(0.1), _state())
    elapsed = time.monotonic() - t0
    renderer.close()
    assert elapsed < 0.5


def test_the_next_due_tick_takes_the_frame_the_peer_was_rendering(daemon, tmp_path):
    renderer, camera = _renderer(daemon, tmp_path, _peer_serving(delay_s=0.3))
    renderer.tick(_tick(0.1), _state())
    renderer.tick(_tick(0.2), _state())
    frames = list(camera.frames)
    renderer.close()
    assert frames == [0.1]


def test_a_peer_that_dies_mid_flight_stops_the_run_naming_it(daemon, tmp_path):
    renderer, _ = _renderer(daemon, tmp_path, _peer_serving(die_after_setup=True))
    with pytest.raises(ConnectionError, match="Kit render peer"):
        for t in (0.1, 0.2):
            renderer.tick(_tick(t), _state())
