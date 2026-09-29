"""The fake Kit peer: the realization of the render peer that starts no container and answers each
``frame`` request with a synthetic frame.

The docker daemon is the system boundary, and the stand-in daemon records any container a run would
start. A test reads a run's frames back from the ``.rrd`` it writes, through Rerun's command-line tool.
"""

import re
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from nexus._src.api.sim import Sim
from nexus._src.peers.kit.fake import KitFake
from nexus._src.peers.kit.runner import KitPeerError
from nexus._src.rendering.link import KitRenderer


def _frames(rrd: str) -> list[tuple[int, int]]:
    """The width and height of every image the recording holds, one entry per frame."""
    out = subprocess.run(
        [sys.executable, "-m", "rerun", "rrd", "print", "-vvv", rrd], capture_output=True, text=True, check=True
    ).stdout
    return [(int(w), int(h)) for w, h in re.findall(r"\{width: (\d+), height: (\d+), pixel_format", out)]


def test_a_run_that_picks_the_fake_kit_peer_starts_no_container_and_gets_frames_at_the_declared_rate(daemon, warp_cpu):
    """A run that picks the fake Kit peer starts no container and gets frames at the declared rate.

    Given `astro_max_fpv`, whose `FpvCam` declares 1280x720 at 24 Hz, a stand-in docker daemon and the
    Kit peer fake, when the run steps 1 s of sim time, 250 ticks of 0.004 s, then the daemon records no
    container, and the camera yields 24 frames, give or take the one in flight at either end, each
    1280x720.
    """
    with Sim("astro_max_fpv", device="cpu", log=True, px4="fake", kit="fake") as sim:
        for _ in range(250):
            sim.step()
    frames = _frames(sim.artifacts()["rrd"])

    assert (daemon.runs, 23 <= len(frames) <= 25, set(frames)) == ([], True, {(1280, 720)}), len(frames)


class _Camera:
    """A sensor riding the link: due at every tick, 2x2 color."""

    path = "/Vehicle/body/Cam"
    output = "color"
    width = height = 2
    body = 0
    mount = np.eye(4)

    def take_due(self, now):
        return True

    def emit(self, arrays, t_shown):
        pass


def _state():
    return SimpleNamespace(body_q=SimpleNamespace(numpy=lambda: np.array([[0, 0, 1, 0, 0, 0, 1.0]])))


def test_a_fake_kit_peer_that_sends_error_ends_the_run_with_that_error():
    """A fake Kit peer that sends `error` ends the run with that error.

    Given a run whose fake Kit peer answers a `frame` request with `error` after 10 frames, when the run
    steps on, then it stops with an error naming the Kit peer and carrying the peer's message.
    """
    peer = KitFake(error_after=10, error="the render product died")
    peer.start()
    renderer = KitRenderer(peer, setup={}, bodies=[(0, "/Vehicle/body")], sensors=[_Camera()])
    renderer.on_physics_ready()

    with pytest.raises(KitPeerError) as raised:
        for i in range(1, 30):
            renderer.tick(SimpleNamespace(sim_time=i * 0.1), _state())
    peer.stop()

    message = str(raised.value)
    assert ("Kit render peer" in message, "the render product died" in message) == (True, True), message
