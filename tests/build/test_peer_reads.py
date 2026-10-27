"""A declared peer takes a component's output over a link the vehicle wires: the peer's schema declares each
input as a relationship ``nexus:inputs:<signal>``, the vehicle targets the prims whose sensors write it, and the
loop hands the peer's end each writer's value, each tick, after the writers. No writer changes. A project's
companion peer takes a camera's frames this way, as the companion computer on the drone takes them over a USB
cable.

The docker daemon and the network are the system boundaries: the stand-in daemon of the pin tests records any
container a run starts, and every fetch fails at once on a cut network. Real builds on the Warp CPU backend of
the fixture vehicle in ``tests/usd/sensor_vehicle.py``, through ``Sim``, with the render stage sent to its fake.
Skipped without newton or pxr.
"""

import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

from typing import ClassVar

import nexus_uav as nx
from nexus_uav._src.peers.px4_sitl.fake import Px4Fake
from nexus_uav._src.vehicle.sensors.rtx.fake import RenderStageFake
from tests.peers import test_pins as pins
from tests.usd import sensor_vehicle as sv

# A declared peer claims a PX4 instance, whose locks and ports are the machine's, so these runs share the one worker.
pytestmark = [pytest.mark.xdist_group("px4_ports"), pytest.mark.usefixtures("warp_cpu")]

# The stand-in docker daemon of the pin tests, with a home folder in the test's folder and the network cut.
daemon = pins.daemon

# The run's length: 1 s of sim time at the default 4 ms tick, over which a 25 Hz camera writes 23 frames, give or take one.
TICKS = 250
# The prims the stand-in peer's inputs target: the pin tests' camera `Cam`, 8 by 6 pixels, a second camera at twice
# its size, so a frame says which camera wrote it, and one quiet Inertial Measurement Unit (IMU), which writes a
# sample every tick on the device.
CAM, CAM_2, IMU_0 = "/vehicle/body/Cam", "/vehicle/body/Cam2", "/vehicle/body/Imu0"
CAMERA_2 = sv.prim(
    "Cam2", "NexusCameraAPI", "int nexus:width = 16\nint nexus:height = 12\nfloat nexus:rate = 25", kind="Camera"
)
IMU = sv.prim("Imu0", "NexusImuAPI", "float nexus:accNoise = 0\nfloat nexus:gyroNoise = 0")


def _companion(refuse: Exception | None = None) -> type:
    """A stand-in class of a project's companion peer, `companion`, on the stand-in peer of the fixture: each value
    its end takes lands in the class's `took` as `(signal, writer, value)`, in the order taken. With `refuse`, the
    first take raises it, as a link whose wire closed. Each call makes a class of its own, for one run.
    """

    class Companion(sv.radio()):
        name = "companion"
        took: ClassVar[list] = []

        def take(self, signal: str, writer: str, value) -> None:
            if refuse is not None:
                raise refuse
            type(self).took.append((signal, writer, value))

        def ports(self) -> dict:
            return {}

    return Companion


def _fly(path: str, registry: nx.Registry, run_dir, ticks: int = TICKS) -> int:
    """Fly the vehicle at `path` in the fixture scene with `registry` for `ticks` ticks through ``Sim``, on the CPU
    device, and return how many it ran.
    """
    n = 0
    with nx.Sim(path, scene=sv.SCENE, device="cpu", registry=registry, run_dir=run_dir) as sim:
        while n < ticks and sim.step():
            n += 1
    return n


def _took(took: list, writer: str) -> list:
    """Each value `writer` gave the peer's end, in the order it came."""
    return [value for _, who, value in took if who == writer]


def _written(kept: list, column: int) -> list[float]:
    """The time of each value a stand-in estimator's `kept` holds in `column`, once, in the order it came."""
    times = []
    for row in kept:
        value = row[column]
        if value is not None and value.time not in times:
            times.append(value.time)
    return times


def test_every_frame_the_camera_writes_reaches_the_peers_link_in_order_with_the_sim_time_it_shows(daemon, tmp_path):
    """Every frame the camera writes reaches the peer's link, in order, with the sim time it shows.

    Given the fixture vehicle with a camera on the render stage's fake and a stand-in peer on its `Companion` scope
    whose `nexus:inputs:camera` targets the camera's prim, when the run steps 1 s of sim time at 25 Hz, then the
    stand-in took the 25 frames the camera wrote, each once, each at the camera's width and height with the sim
    time it shows, in the order written. A stand-in estimator reading `camera` says what the camera wrote.
    """
    peer = _companion()
    reader = sv.estimator(nx.Signal("camera", nx.Image))
    path = sv.vehicle(tmp_path, pins.CAMERA, root=sv.companion(inputs={"camera": CAM}), estimator="")
    registry = sv.components(StandInPeerAPI=peer, StandInEstimatorAPI=reader, ovrtx=RenderStageFake)

    _fly(path, registry, tmp_path / "run")

    frames = _took(peer.took, "cam")
    seen = (
        len(frames) == 25,
        all(frame.pixels.shape[:2] == (6, 8) for frame in frames),
        [frame.time for frame in frames] == _written(reader.kept, 1),
    )
    assert seen == (True, True, True), (len(frames), peer.took[:3], _written(reader.kept, 1)[:3])


def test_each_of_two_cameras_feeds_the_peer_told_apart_by_its_sensors_name(daemon, tmp_path):
    """Each of two cameras feeds the peer, told apart by its sensor's name.

    Given the fixture vehicle with two cameras, `Cam` at 8 by 6 pixels and `Cam2` at 16 by 12, on the render stage's
    fake and a stand-in peer whose `nexus:inputs:camera` targets both, when the run steps 1 s, then the stand-in took
    each camera's frames under that camera's name, `cam` and `cam2`, 25 each, each once, and neither
    camera's frames under the other's: every frame under `cam` is 8 by 6 and every frame under `cam2` 16 by 12.
    """
    peer = _companion()
    path = sv.vehicle(tmp_path, pins.CAMERA + CAMERA_2, root=sv.companion(inputs={"camera": [CAM, CAM_2]}))
    registry = sv.components(StandInPeerAPI=peer, ovrtx=RenderStageFake)

    _fly(path, registry, tmp_path / "run")

    cam, cam2 = _took(peer.took, "cam"), _took(peer.took, "cam2")
    seen = (
        {writer for _, writer, _ in peer.took},
        len(cam) == 25,
        len(cam2) == 25,
        all(frame.pixels.shape[:2] == (6, 8) for frame in cam),
        all(frame.pixels.shape[:2] == (12, 16) for frame in cam2),
    )
    assert seen == ({"cam", "cam2"}, True, True, True, True), (len(cam), len(cam2))


def test_a_peer_wired_to_a_signal_a_device_stage_writes_reads_it_each_tick(daemon, tmp_path):
    """A peer wired to a signal a device stage writes reads it each tick.

    Given the fixture vehicle with a quiet IMU and a stand-in peer whose `nexus:inputs:imu` targets the IMU's prim,
    when the run steps 1 s, then the stand-in took one sample per tick the IMU wrote, 250 give or take the start's,
    each with the sim time of its tick. A stand-in estimator reading `imu` says which ticks the IMU wrote.
    """
    peer = _companion()
    reader = sv.estimator(nx.Signal("imu", nx.ImuSample, shape=(1,)))
    path = sv.vehicle(tmp_path, IMU, root=sv.companion(inputs={"imu": IMU_0}), estimator="")
    registry = sv.components(StandInPeerAPI=peer, StandInEstimatorAPI=reader)

    _fly(path, registry, tmp_path / "run")

    times = [float(value.time) for signal, writer, value in peer.took if (signal, writer) == ("imu", "imu0")]
    seen = (250 <= len(times) <= 251, times == [float(sample.time) for _, sample in reader.kept])
    assert seen == (True, True), (len(times), peer.took[:3], reader.kept[:3])


def test_a_vehicle_whose_declared_peers_wire_no_input_flies_as_today(daemon, tmp_path):
    """A vehicle whose declared peers wire no input flies as today.

    Given the fixture vehicle with a camera on the render stage's fake and the PX4 Software In The Loop (SITL) peer
    on its fake, which wires no input, and a stand-in estimator reading `camera`, when the run steps 1 s, then the
    camera's 25 frames reach their reader, each at the camera's width and height.
    """
    reader = sv.estimator(nx.Signal("camera", nx.Image))
    path = sv.vehicle(tmp_path, pins.CAMERA, px4=True, estimator="")
    registry = sv.components(NexusPx4SitlAPI=Px4Fake, StandInEstimatorAPI=reader, ovrtx=RenderStageFake)

    ran = _fly(path, registry, tmp_path / "run")

    frames = _written(reader.kept, 1)
    seen = (ran, len(frames) == 25, all(f.pixels.shape[:2] == (6, 8) for _, f in reader.kept if f is not None))
    assert seen == (TICKS, True, True), len(frames)


def test_a_frame_the_link_cant_take_fails_the_run_and_the_error_names_the_peer(daemon, tmp_path):
    """A frame the link can't take fails the run, and the error names the peer.

    Given the fixture vehicle with a camera on the render stage's fake and a stand-in peer wired to it whose end raises
    `wire closed` on its first frame, when the run steps, then `Sim` raises a `RunError` whose message names the
    stand-in, `companion`, and the run's peers stopped.
    """
    peer = _companion(refuse=OSError("wire closed"))
    path = sv.vehicle(tmp_path, pins.CAMERA, root=sv.companion(inputs={"camera": CAM}))
    registry = sv.components(StandInPeerAPI=peer, ovrtx=RenderStageFake)

    with pytest.raises(nx.RunError) as raised:
        _fly(path, registry, tmp_path / "run")

    assert ("companion" in str(raised.value), peer.events) == (True, ["start", "stop"]), str(raised.value)
