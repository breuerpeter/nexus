"""The fake Kit peer: the class a peer mapping sends the Kit peer to, which starts no container and answers
each ``frame`` request with a synthetic frame.

The docker daemon is the system boundary, and the stand-in daemon records any container a run would
start. A test reads a run's frames back from the ``.rrd`` it writes, through Rerun's reader.
"""

import functools
import io
import time

from PIL import Image
from rerun.experimental import RrdReader

import nexus_sim._src.build.launch as launch_mod
from nexus_sim._src.config import LaunchConfig
from nexus_sim._src.peers.kit.fake import KitFake
from nexus_sim._src.peers.kit.runner import KitPeerError
from nexus_sim._src.peers.px4_sitl.fake import Px4Fake


def _frames(rrd: str, camera: str) -> list[tuple[int, int]]:
    """The width and height of every frame the recording holds for ``camera``, one entry per frame.

    The logger stores a camera frame as an encoded image, so this decodes each one for its size.
    """
    sizes = []
    for chunk in RrdReader(rrd).stream():
        batch = chunk.to_record_batch()
        if chunk.entity_path == f"/sim/vehicle/sensors/{camera}" and "EncodedImage:blob" in batch.schema.names:
            for cell in batch.column("EncodedImage:blob").to_pylist():
                sizes.append(Image.open(io.BytesIO(bytes(cell[0]))).size)
    return sizes


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
    launch = LaunchConfig.from_dict(
        {"vehicle": "astro_max_fpv", "scene": "empty", "runtime": {"device": "cpu"}, "output": {"log": True}}
    )
    loop = launch_mod.build_from_launch(launch, preroll_timeout=10.0, peers={"px4_sitl": Px4Fake, "kit": KitFake})

    ticks = 0
    while ticks < 250 and loop.step():  # stepping an ended run starts a new one
        ticks += 1
    loop.close()
    frames = _frames(loop.logger.rrd_path, "fpvcam")

    assert (daemon.runs, 23 <= len(frames) <= 25, set(frames)) == ([], True, {(1280, 720)}), len(frames)


def test_a_fake_kit_peer_that_sends_error_ends_the_run_with_that_error(daemon, warp_cpu, monkeypatch):
    """A fake Kit peer that sends `error` ends the run with that error.

    Given a run of `astro_max_fpv` whose fake Kit peer answers a `frame` request with `error` after 10
    frames, when the run steps on, then it stops with an error naming the Kit peer and carrying the
    peer's message.
    """
    # The network is a boundary: a PX4 fetch this run must not make fails at once on an unreachable proxy.
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.setenv(var, "http://127.0.0.1:9")
    kit = functools.partial(KitFake, error_after=10, error="the render product died")
    launch = LaunchConfig.from_dict({"vehicle": "astro_max_fpv", "scene": "empty", "runtime": {"device": "cpu"}})
    loop = launch_mod.build_from_launch(launch, preroll_timeout=10.0, peers={"px4_sitl": Px4Fake, "kit": kit})

    ticks = 0
    try:
        while ticks < 1000 and loop.step():  # 10 frames at 24 Hz take about 105 ticks of 0.004 s
            ticks += 1
        message = f"no error after {ticks} ticks"
    except KitPeerError as exc:
        message = str(exc)

    assert ("Kit render peer" in message, "the render product died" in message) == (True, True), message


def test_the_kit_fake_stops_at_once_when_no_run_connected_to_it():
    """The Kit fake stops at once when no run connected to it."""
    fake = KitFake()
    fake.start()
    began = time.monotonic()
    fake.stop()
    assert time.monotonic() - began < 1.0
