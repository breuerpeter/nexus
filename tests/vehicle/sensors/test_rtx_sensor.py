"""An RTX sensor writes each frame or scan the Kit peer hands it to its output signal, stamped with the sim time
it shows. Each sensor here sits on a prim of an in-memory stage and takes its frames by hand, so no peer runs.
Skipped without pxr.
"""

import numpy as np
import pytest

pytest.importorskip("pxr")

from nexus_sim._src.core.interfaces import SensorRun
from nexus_sim._src.vehicle.sensors.rtx_camera import RtxCameraSensor
from nexus_sim._src.vehicle.sensors.rtx_lidar import RtxLidarSensor
from nexus_sim._src.vehicle.sensors.rtx_thermal import RtxThermalSensor
from tests.usd import sensor_vehicle as sv

pytestmark = pytest.mark.usefixtures("warp_cpu")

WIDTH, HEIGHT = 8, 6


@pytest.fixture
def run():
    """The run's values for a sensor on the camera prim `/vehicle/body/Cam` of an in-memory stage, which lives
    for the test.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/vehicle").GetPrim())
    yield SensorRun(prim=UsdGeom.Camera.Define(stage, "/vehicle/body/Cam").GetPrim())


@pytest.mark.parametrize(
    ("sensor", "arrays", "content"),
    [
        (
            lambda run: RtxCameraSensor(run, width=WIDTH, height=HEIGHT),
            {"color": np.full((HEIGHT, WIDTH, 3), 7, dtype=np.uint8)},
            lambda frame: (frame.pixels.shape, int(frame.pixels.max())) == ((HEIGHT, WIDTH, 3), 7),
        ),
        (
            lambda run: RtxThermalSensor(run, width=WIDTH, height=HEIGHT),
            {
                "radiance": np.zeros((HEIGHT, WIDTH), dtype=np.float32),
                "depth": np.ones((HEIGHT, WIDTH), dtype=np.float32),
            },
            lambda frame: (frame.pixels.shape, frame.pixels.dtype) == ((HEIGHT, WIDTH, 3), np.uint8),
        ),
        (
            RtxLidarSensor,
            {"points": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)},
            lambda scan: scan.points.tolist() == [[1.0, 2.0, 3.0]],
        ),
    ],
    ids=["camera", "thermal_camera", "lidar"],
)
def test_an_rtx_sensor_writes_each_frame_to_its_signal_with_the_sim_time_it_shows(run, sensor, arrays, content):
    """An RTX sensor writes each frame or scan to its output signal, with the sim time it shows.

    Given a camera, a thermal camera or a lidar, wired alone, when the render link hands it a frame that shows
    sim time 0.5 s, then its output signal holds that frame, stamped 0.5 s: the camera's pixels, the thermal
    camera's 8-bit image at the frame's size, or the lidar's points.
    """
    rtx = sv.wired(sensor(run))
    rtx.emit(arrays, 0.5)
    frame = rtx.out.read()

    assert (frame.time, content(frame)) == (0.5, True)
