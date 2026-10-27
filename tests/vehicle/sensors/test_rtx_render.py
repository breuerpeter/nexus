"""The RTX sensors render in process: a run of the fixture vehicle with a camera, a thermal camera or a lidar gets
its frames from the run's own renderer, and asks the docker daemon for no container.

Each test builds the fixture vehicle on CUDA in `render_scene.usda`, a gray ground with a red square at the
origin under a sun and a sky, and reads the sensor's signal after every tick. The docker client fails a test that
asks it for a container. A camera looks down: the base body's frame is Forward Right Down (FRD), so a prim rolled
180 degrees about the body's forward axis looks along the body's +Z, which points at the ground.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

from nexus_uav._src.core.errors import RunError
from tests.usd import sensor_vehicle as sv

pytestmark = [pytest.mark.gpu, pytest.mark.usefixtures("no_docker")]

SCENE = str(Path(__file__).with_name("render_scene.usda"))
WIDTH, HEIGHT = 320, 240
# A camera rolled to face the ground.
LOOK_DOWN = 'float xformOp:rotateX = 180\nuniform token[] xformOpOrder = ["xformOp:rotateX"]'
CAMERA = sv.prim(
    "Cam", "NexusCameraAPI", f"int nexus:width = {WIDTH}\nint nexus:height = {HEIGHT}\nfloat nexus:rate = 25\n{LOOK_DOWN}",
    kind="Camera",
)  # fmt: skip
THERMAL = sv.prim(
    "Ir", "NexusThermalCameraAPI",
    f"int nexus:width = {WIDTH}\nint nexus:height = {HEIGHT}\nfloat nexus:rate = 25\n{LOOK_DOWN}",
    kind="Camera",
)  # fmt: skip
# A lidar 2 m over the base body, whose -Z is up, with the generic lidar schema and no frame rate of its own.
LIDAR = (
    'def OmniLidar "Scan" (\n    prepend apiSchemas = ["NexusLidarAPI", "OmniSensorGenericLidarCoreAPI"]\n)\n{\n'
    "    float nexus:rate = 10\n"
    "    double3 xformOp:translate = (0, 0, -2)\n"
    '    uniform token[] xformOpOrder = ["xformOp:translate"]\n}\n'
)


def _fly(vehicle, signal, ticks, *, fall_from=None, **runtime):
    """Fly the fixture vehicle at `vehicle` for `ticks` ticks of 4 ms. Return each new value of `signal`, in order,
    with the sim time of the tick it arrived on, then the value it holds after the run closes, if new, with the time
    of the last tick; a value is new when its stamp is. Return beside them the error that ended a run its build
    refused, which renders nothing, or "".
    """
    try:
        loop = sv.build(vehicle, scene=SCENE, device="cuda", fall_from=fall_from, runtime=runtime)
    except (RunError, ValueError) as exc:  # a launch that refuses a runtime value raises ValueError
        return [], str(exc)
    got, seen, now = [], set(), 0.0
    for _ in range(ticks):
        if not loop.step():
            break
        now = loop.clock.now().sim_time
        value = loop.signals[signal].read()
        if value is not None and value.time not in seen:
            seen.add(value.time)
            got.append((now, value))
    loop.close()
    value = loop.signals[signal].read()
    if value is not None and value.time not in seen:
        got.append((now, value))
    return got, ""


def _heights(vehicle, ticks, *, fall_from):
    """The base body's height after each of `ticks` ticks of a flight of the fixture vehicle at `vehicle`, by the
    tick's sim time.
    """
    loop = sv.build(vehicle, scene=SCENE, device="cuda", fall_from=fall_from)
    heights = {}
    for _ in range(ticks):
        if not loop.step():
            break
        heights[loop.clock.now().sim_time] = float(loop.signals["vehicle/body/body"].read().position[2])
    loop.close()
    return heights


def _difference(a, b) -> float:
    """The mean absolute difference of two frames' pixels, in 8-bit levels."""
    return float(np.abs(a.pixels.astype(np.float64) - b.pixels.astype(np.float64)).mean())


def test_a_run_of_a_camera_vehicle_renders_its_frames_in_process_at_the_declared_rate(tmp_path):
    """A run of a camera vehicle renders its frames in process: they reach the camera's signal at its declared rate,
    and no Kit container starts.

    Given a camera at 25 Hz and 320x240 on the fixture vehicle, resting in a scene with geometry, and a docker
    daemon the run must not ask, when the run flies 1 s, 250 ticks of 4 ms, then the camera's signal holds 25
    frames, each 320x240 and none blank, its pixels not all one value.
    """
    frames, error = _fly(sv.vehicle(tmp_path, CAMERA), "vehicle/sensors/cam/camera", 250)
    sizes = [(frame.pixels.shape, bool(frame.pixels.std() > 5.0)) for _, frame in frames]

    assert sizes == [((HEIGHT, WIDTH, 3), True)] * 25, error


def test_by_default_a_frame_shows_the_tick_that_asked_for_it(tmp_path):
    """By default a frame shows the tick that asked for it, stamped with that tick's time.

    Given a camera at 25 Hz looking down from the fixture vehicle as it falls from 2 m, when the run renders with
    the default runtime, then every frame reaches the signal on the tick its stamp names, and its sixth frame, at
    0.204 s, matches within 3 levels a frame rendered with the vehicle starting at the height it fell to by then.
    """
    vehicle = sv.vehicle(tmp_path, CAMERA)
    frames, error = _fly(vehicle, "vehicle/sensors/cam/camera", 60, fall_from=(0.0, 0.0, 2.0))
    on_time = bool(frames) and all(t == frame.time for t, frame in frames)
    matches = False
    if len(frames) > 5:
        shown, frame = frames[5]
        height = _heights(vehicle, 60, fall_from=(0.0, 0.0, 2.0))[shown]
        still, _ = _fly(vehicle, "vehicle/sensors/cam/camera", 1, fall_from=(0.0, 0.0, height))
        matches = bool(still) and _difference(frame, still[0][1]) <= 3.0

    assert (on_time, matches) == (True, True), error


def test_a_pipelined_run_gets_each_frame_one_due_tick_later_stamped_with_the_tick_it_shows(tmp_path):
    """With `pipelined_render` on, a due tick's frame arrives at the next due tick, stamped with the tick it shows,
    and the last frame reaches the signal at close.

    Given a camera at 25 Hz looking down from the fixture vehicle as it falls from 2 m, when the same 0.4 s flies
    once synchronous and once pipelined, then the two signals hold the same stamps, the pipelined run's last frame
    included, each pipelined frame arrives one due tick, 40 ms, after its stamp, and each pair of frames matches
    within 3 levels.
    """
    vehicle = sv.vehicle(tmp_path, CAMERA)
    synchronous, error = _fly(vehicle, "vehicle/sensors/cam/camera", 100, fall_from=(0.0, 0.0, 2.0))
    pipelined, _ = _fly(vehicle, "vehicle/sensors/cam/camera", 100, fall_from=(0.0, 0.0, 2.0), pipelined_render=True)
    stamps = [frame.time for _, frame in pipelined]
    late = {round(t - frame.time, 6) for t, frame in pipelined[:-1]}
    pairs = list(zip(synchronous, pipelined, strict=True)) if len(synchronous) == len(pipelined) else []
    matches = bool(pairs) and max(_difference(a, b) for (_, a), (_, b) in pairs) <= 3.0

    assert (bool(stamps) and stamps == [f.time for _, f in synchronous], late == {0.04}, matches) == (
        True,
        True,
        True,
    ), error


def test_a_thermal_cameras_frames_reach_its_signal_at_its_declared_rate(tmp_path):
    """A thermal camera's frames reach its signal at its declared rate.

    Given a thermal camera at 25 Hz on the fixture vehicle, resting in a scene with geometry, and a docker daemon
    the run must not ask, when the run flies 1 s, 250 ticks of 4 ms, then the thermal camera's signal holds 25
    frames.
    """
    frames, error = _fly(sv.vehicle(tmp_path, THERMAL), "vehicle/sensors/ir/thermal_camera", 250)

    assert len(frames) == 25, error


def test_a_lidar_scans_once_per_period_of_its_rate_in_the_world_frame_with_no_frame_rate_of_its_own(tmp_path):
    """A lidar completes one scan per period of its declared rate, even when its file authors no
    `omni:sensor:frameRate`. Its points are in the world frame, as today.

    Given a lidar at 10 Hz 2 m over the fixture vehicle's base body, resting on flat ground, whose prim authors no
    frame rate, when the run flies 1 s, 250 ticks of 4 ms, then the signal holds 10 scans, none empty, and every
    point's height is within 5 cm of the ground's, 0 m.
    """
    scans, error = _fly(sv.vehicle(tmp_path, LIDAR), "vehicle/sensors/scan/lidar", 250)
    held = [(len(scan.points) > 0, bool(np.all(np.abs(scan.points[:, 2]) < 0.05))) for _, scan in scans]

    assert held == [(True, True)] * 10, error
