"""The fake Kit peer: the class a peer mapping sends the Kit peer to, which starts no container and answers
each ``frame`` request with a synthetic frame.

The docker daemon is the system boundary, and the stand-in daemon records any container a run would
start. A test reads a run's frames back from the ``.rrd`` it writes, through Rerun's command-line tool.
"""

import re
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import nexus._src.build.launch as launch_mod
from nexus._src.config import LaunchConfig
from nexus._src.peers.kit.fake import KitFake
from nexus._src.peers.kit.runner import KitPeerError
from nexus._src.peers.px4_sitl.fake import Px4Fake
from nexus._src.rendering.link import KitRenderer


def _frames(rrd: str) -> list[tuple[int, int]]:
    """The width and height of every image the recording holds, one entry per frame."""
    out = subprocess.run(
        [sys.executable, "-m", "rerun", "rrd", "print", "-vvv", rrd], capture_output=True, text=True, check=True
    ).stdout
    return [(int(w), int(h)) for w, h in re.findall(r"\{width: (\d+), height: (\d+), pixel_format", out)]


def test_a_run_whose_peer_mapping_sends_the_kit_peer_to_its_fake_starts_no_container_and_gets_frames_at_the_declared_rate(
    daemon, warp_cpu, monkeypatch
):
    """A run whose peer mapping sends the Kit peer to its fake starts no container and gets frames at the
    declared rate.

    Given `astro_max_fpv`, whose `FpvCam` declares 1280x720 at 24 Hz, a stand-in docker daemon and a
    peer mapping that sends the Kit peer to its fake, when the run steps 1 s of sim time, 250 ticks of 0.004 s,
    then the daemon records no container, and the camera yields 24 frames, give or take the one in flight
    at either end, each 1280x720. The PX4 peer maps to its fake too, so nothing else starts a container.
    """
    # The network is a boundary: a PX4 fetch this run must not make fails at once on an unreachable proxy.
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.setenv(var, "http://127.0.0.1:9")
    launch = LaunchConfig.from_dict({"vehicle": "astro_max_fpv", "runtime": {"device": "cpu"}, "output": {"log": True}})
    loop = launch_mod.build_from_launch(launch, preroll_timeout=10.0, peers={"px4_sitl": Px4Fake, "kit": KitFake})

    ticks = 0
    while ticks < 250 and loop.step():  # stepping an ended run starts a new one
        ticks += 1
    loop.close()
    frames = _frames(loop.logger.rrd_path)

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

    Given a fake Kit peer that answers a `frame` request with `error` after 10 frames, when the run steps
    on, then it stops with an error naming the Kit peer and carrying the peer's message.
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
