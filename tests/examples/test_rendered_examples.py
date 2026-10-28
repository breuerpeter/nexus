"""An example that self-assembles its loop renders the RTX sensors its vehicle declares: the loop hands each the
render stage at connect. Real build on the Warp CPU backend against the render stage's fake. Skipped without newton
or pxr.
"""

import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

from nexus_uav._src.build.launch import resolve_vehicle_usd
from nexus_uav._src.config import LaunchConfig
from nexus_uav._src.core.registry import Registry
from nexus_uav._src.vehicle.sensors.rtx.fake import RenderStageFake
from nexus_uav.examples.controllers.pid.assembly import build_pid_loop
from tests.usd import sensor_vehicle as sv

pytestmark = pytest.mark.usefixtures("warp_cpu")

# A camera at 25 Hz on the base body, small so the fake's frames cost little.
CAMERA = sv.prim(
    "Cam", "NexusCameraAPI", "int nexus:width = 8\nint nexus:height = 6\nfloat nexus:rate = 25", kind="Camera"
)


def test_the_pid_example_on_a_camera_vehicle_hands_its_camera_frames_from_the_render_stage(tmp_path):
    """The Proportional Integral Derivative (PID) example on a vehicle with a camera hands the camera its frames from
    the render stage.

    Given the fixture vehicle with a camera at 25 Hz and the render stage sent to its fake, when the example's
    loop steps 50 ticks, 0.2 s of sim time, then the camera's signal holds a frame.
    """
    cfg = sv.scenario()
    cfg["physics"]["force_cpu"] = True
    vehicle_usd, _ = resolve_vehicle_usd(LaunchConfig().set_vehicle(sv.vehicle(tmp_path, CAMERA)).set_scene(sv.SCENE))
    loop = build_pid_loop(
        cfg, goal_w=(0.0, 0.0, 1.5), max_steps=50, vehicle_usd=vehicle_usd,
        components=Registry({"ovrtx": RenderStageFake}),
    )  # fmt: skip
    (camera,) = [s for s in loop.sensors if s.requires]
    loop.run()

    assert camera.out.read() is not None
