"""Where a camera's, a thermal camera's and a lidar's rows sit in a run's recording.

Real end-to-end on the Warp CPU backend against the Kit peer's fake: the tests read the ``.rrd`` a
run wrote through Rerun's reader. Most read the session's one recorded flight, which
``tests/conftest.py`` flies with the cameras `fpvcam`, `a` and `b` and the thermal camera `ir`, all
on the base body. Skipped if rerun or newton are missing.
"""

import functools
import logging

import pytest

pytest.importorskip("rerun")
pytest.importorskip("newton")

from tests.conftest import fly_recorded
from tests.usd import sensor_vehicle as sv

# A camera, `FpvCam`, which the run names `fpvcam`.
FPVCAM = sv.prim("FpvCam", "NexusCameraAPI", kind="Camera")
# A lidar, `Scan`, which the run names `scan`.
LIDAR = sv.prim("Scan", "NexusLidarAPI", "float nexus:rate = 25", kind="OmniLidar")


def test_two_instances_of_one_class_write_under_their_own_names(recorded_flight, rrd_rows):
    """Two instances of one class write under their own names.

    Given a flight with two cameras of one class, `a` and `b`, and the Kit peer's fake, when the
    recorded flight steps past one frame of each, then the `.rrd` holds image rows at
    `/sim/vehicle/sensors/a/camera` and at `/sim/vehicle/sensors/b/camera`.
    """
    framed = sorted(
        {
            entity
            for entity, _, columns in rrd_rows(recorded_flight)
            if "EncodedImage:blob" in columns and entity.split("/")[-2] in ("a", "b")
        }
    )

    assert framed == ["/sim/vehicle/sensors/a/camera", "/sim/vehicle/sensors/b/camera"]


def test_a_cameras_or_thermal_cameras_frustum_sits_at_its_entity_and_its_frames_at_its_signals_row(
    recorded_flight, rrd_rows
):
    """A camera's or thermal camera's frustum sits at `sim/vehicle/sensors/<n>`, and its frames at the row
    named after its signal below it.

    Given a flight with the camera `fpvcam`, the thermal camera `ir` and the Kit peer's fake, when the
    recorded flight steps past one frame of each, then each sensor's entity holds one `Pinhole` row, its
    image rows sit at `.../fpvcam/camera` and `.../ir/thermal_camera`, and the `.rrd` holds no entity
    under `/vehicle/body/cameras/` or `/cameras/`.
    """
    rows = rrd_rows(recorded_flight)
    held = {}
    for sensor, signal in (("/sim/vehicle/sensors/fpvcam", "camera"), ("/sim/vehicle/sensors/ir", "thermal_camera")):
        at_sensor = [columns for entity, _, columns in rows if entity == sensor]
        at_row = [columns for entity, _, columns in rows if entity == f"{sensor}/{signal}"]
        frames = sum(len(c["EncodedImage:blob"]) for c in at_row if "EncodedImage:blob" in c)
        misplaced = sum(len(c["EncodedImage:blob"]) for c in at_sensor if "EncodedImage:blob" in c)
        frustums = sum(len(c["Pinhole:image_from_camera"]) for c in at_sensor if "Pinhole:image_from_camera" in c)
        held[sensor] = (frames > 0, misplaced, frustums)
    old = sorted({entity for entity, _, _ in rows if entity.startswith(("/vehicle/body/cameras/", "/cameras/"))})

    assert (held, old) == (
        {"/sim/vehicle/sensors/fpvcam": (True, 0, 1), "/sim/vehicle/sensors/ir": (True, 0, 1)},
        [],
    )


def test_a_camera_or_thermal_camera_rides_the_body_through_one_static_transform_with_no_transform_row_per_tick(
    recorded_flight, rrd_rows
):
    """A camera or thermal camera rides the body through one static transform, with no transform row per tick.

    Given a flight with the camera `fpvcam`, the thermal camera `ir` and the Kit peer's fake, when the
    recorded flight steps 100 ticks, then each sensor's entity holds exactly one transform row, static,
    whose `parent_frame` names the frame of `/sim/vehicle/body`.
    """
    transforms: dict[str, list] = {"/sim/vehicle/sensors/fpvcam": [], "/sim/vehicle/sensors/ir": []}
    for entity, static, columns in rrd_rows(recorded_flight):
        if entity in transforms and "Transform3D:translation" in columns:
            parents = columns["Transform3D:parent_frame"].to_pylist() if "Transform3D:parent_frame" in columns else []
            for row in range(len(columns["Transform3D:translation"])):
                transforms[entity].append((static, parents[row][0] if parents else None))

    assert transforms == {
        "/sim/vehicle/sensors/fpvcam": [(True, "tf#/sim/vehicle/body")],
        "/sim/vehicle/sensors/ir": [(True, "tf#/sim/vehicle/body")],
    }


def test_a_camera_whose_frustum_fails_to_log_still_records_its_frames_at_its_own_path(
    tmp_path, monkeypatch, caplog, rrd_rows
):
    """A camera whose frustum fails to log still records its frames at its own path.

    Given the fixture vehicle with the camera `fpvcam`, whose `Pinhole` row raises when logged, when a recorded run steps past
    one frame, then the image rows sit at `/sim/vehicle/sensors/fpvcam/camera` and the log holds one warning.
    """
    import rerun as rr

    log = rr.log

    def fail_on_frustum(entity, *components, **kwargs):
        # Rerun is the boundary: it refuses the frustum and takes every other row.
        if any(isinstance(component, rr.Pinhole) for component in components):
            raise RuntimeError("the frustum failed to log")
        return log(entity, *components, **kwargs)

    monkeypatch.setattr(rr, "log", fail_on_frustum)
    with caplog.at_level(logging.WARNING):
        rrd = fly_recorded(tmp_path, sv.vehicle(tmp_path, FPVCAM), scene=sv.SCENE, ticks=30)
    framed = sorted({entity for entity, _, columns in rrd_rows(rrd) if "EncodedImage:blob" in columns})
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]

    assert (framed, len(warnings)) == (["/sim/vehicle/sensors/fpvcam/camera"], 1), warnings


def test_a_lidars_points_sit_at_its_signals_row_in_the_world_frame_as_today_with_no_transform_row(tmp_path, rrd_rows):
    """A lidar's points sit at `sim/vehicle/sensors/<n>/lidar`, the row named after its signal, in the world
    frame as today, with no transform row on the way.

    Given the fixture vehicle with the lidar `scan` and the Kit peer's fake scanning one known world point,
    `(1, 2, 3)`, when a recorded run steps past one scan, then the points row at
    `/sim/vehicle/sensors/scan/lidar` holds that world point, neither it nor the sensor's entity holds a
    transform row, and the `.rrd` holds no entity under `/lidar/`.
    """
    from nexus_sim._src.peers.kit.fake import KitFake

    kit = functools.partial(KitFake, points=[[1.0, 2.0, 3.0]])
    rows = rrd_rows(fly_recorded(tmp_path, sv.vehicle(tmp_path, LIDAR), scene=sv.SCENE, ticks=30, kit=kit))
    at_row = [columns for entity, _, columns in rows if entity == "/sim/vehicle/sensors/scan/lidar"]
    at_sensor = [columns for entity, _, columns in rows if entity == "/sim/vehicle/sensors/scan"]
    points = {
        tuple(point)
        for c in at_row
        if "Points3D:positions" in c
        for scan in c["Points3D:positions"].to_pylist()
        for point in scan
    }
    transforms = sorted({name for c in (*at_row, *at_sensor) for name in c if name.startswith("Transform3D:")})
    old = sorted({entity for entity, _, _ in rows if entity.startswith("/lidar/")})

    assert (points, transforms, old) == ({(1.0, 2.0, 3.0)}, [], [])
