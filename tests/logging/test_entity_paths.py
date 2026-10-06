"""A recorded row's entity path names its process, its component and its instance.

Real end-to-end on the Warp CPU backend: the tests read the ``.rrd`` a run wrote, and the layout it
stores, through Rerun's reader. Most read the session's one recorded flight of the shipped camera
vehicle, which ``tests/conftest.py`` flies against the Kit peer's fake. Skipped if rerun or newton
are missing.
"""

import os

import pytest

pytest.importorskip("rerun")
pytest.importorskip("newton")

from tests.conftest import fly_recorded
from tests.usd import sensor_vehicle as sv

# Rerun's kind number for the EntityPath column of a log view.
ENTITY_PATH_COLUMN = 1


def test_every_row_the_sim_writes_sits_under_sim(recorded_flight, rrd_rows):
    """Every row the sim writes sits under `sim/`.

    Given a recorded flight of a vehicle with mesh shapes, a ground plane, rotor joints, an Inertial
    Measurement Unit (IMU), cameras, a stand-in controller and an operator, when it ends, then every
    entity path in the `.rrd` starts with `/sim/`. Rerun writes `/__properties` itself.
    """
    entities = {entity for entity, _, _ in rrd_rows(recorded_flight)}

    assert sorted(e for e in entities if not e.startswith(("/sim/", "/__"))) == []


def test_a_log_record_lands_at_sim_logs_module_and_its_text_is_the_bare_message(tmp_path, rrd_rows):
    """A log record lands at `sim/logs/<module>`, and its text is the bare message.

    Given a recording, when this module calls `nexus_sim.logger.info("hello")`, then the `.rrd` holds a
    `TextLog` row at `/sim/logs/test_entity_paths` whose text is `hello`.
    """
    import nexus_sim as nx
    from nexus_sim._src.logging import Logger

    rrd = str(tmp_path / "log.rrd")
    sink = Logger(model=None, serve=False, record_to_rrd=rrd)
    nx.logger.info("hello")
    sink.close()
    texts = [
        text[0]
        for entity, _, columns in rrd_rows(rrd)
        if entity == "/sim/logs/test_entity_paths"
        for text in columns["TextLog:text"].to_pylist()
    ]

    assert texts == ["hello"]


def test_the_runs_settings_profile_and_rtf_rows_sit_under_sim_run(tmp_path, rrd_rows):
    """The run's settings, profile and RTF rows sit under `sim/run/`.

    Given a recording with settings, when the run logs its Real Time Factor (RTF) and its profile and
    ends, then the `.rrd` holds `/sim/run/settings`, `/sim/run/profile` and `/sim/run/rtf`.
    """
    from nexus_sim._src.logging import Logger

    rrd = str(tmp_path / "run.rrd")
    sink = Logger(model=None, serve=False, record_to_rrd=rrd, settings={"seed": 42})
    sink.set_time(0.5)
    sink.log_rtf(1.23)
    sink.log_profile({"control_steps": 100, "rtf": 1.2, "profile": {"tick_p50_ms": 0.04}})
    sink.close()
    documents = sorted({entity for entity, _, columns in rrd_rows(rrd) if "TextDocument:text" in columns})

    assert documents == ["/sim/run/profile", "/sim/run/rtf", "/sim/run/settings"]


def test_the_vehicle_bodys_pose_sits_at_sim_vehicle_body(recorded_flight, rrd_rows):
    """The vehicle body's pose sits at `sim/vehicle/body`.

    Given a recorded flight of 100 ticks of 0.004 s, which the Logger's 50 Hz log rate makes 20 logged
    ticks, when it ends, then `/sim/vehicle/body` holds one transform row per logged tick.
    """
    poses = sum(
        len(columns["Transform3D:translation"])
        for entity, _, columns in rrd_rows(recorded_flight)
        if entity == "/sim/vehicle/body" and "Transform3D:translation" in columns
    )

    assert poses == 20


def test_a_debug_run_writes_one_frame_per_body_at_sim_vehicle_body_label(tmp_path, rrd_rows):
    """A debug run writes one frame per body at `sim/vehicle/body/<label>`.

    Given a debug recorded run of the local fixture vehicle, whose bodies are `body` and four rotors, when
    it ends, then the `.rrd` holds a frame at `/sim/vehicle/body/<label>` for each body and no entity
    at a Universal Scene Description (USD) prim path.
    """
    rows = rrd_rows(fly_recorded(tmp_path, sv.vehicle(tmp_path), scene=sv.SCENE, ticks=20, debug=True))
    frames = sorted({entity for entity, _, columns in rows if "TransformAxes3D:axis_length" in columns})
    at_prim_paths = sorted({entity for entity, _, _ in rows if entity.startswith(sv.ROOT)})

    assert (frames, at_prim_paths) == (
        [
            "/sim/vehicle/body/body",
            "/sim/vehicle/body/rotor_1",
            "/sim/vehicle/body/rotor_2",
            "/sim/vehicle/body/rotor_3",
            "/sim/vehicle/body/rotor_4",
        ],
        [],
    )


def test_nvidia_newtons_scene_rows_sit_under_sim_model_and_sim_geometry(recorded_flight, rrd_rows):
    """NVIDIA Newton's scene rows sit under `sim/model/` and `sim/geometry/`.

    Given a recorded flight of a vehicle with mesh shapes over a ground plane, when it ends, then the
    `.rrd` holds its shapes under `/sim/model/` and its mesh and plane under `/sim/geometry/`, and no
    entity under `/model/` or `/geometry/`.
    """
    entities = {entity for entity, _, _ in rrd_rows(recorded_flight)}
    scene = {"/sim/model/shapes/shape_0", "/sim/model/shapes/shape_1", "/sim/geometry/mesh_1", "/sim/geometry/plane_0"}

    assert (sorted(scene - entities), sorted(e for e in entities if e.startswith(("/model/", "/geometry/")))) == (
        [],
        [],
    )


def test_the_scene_view_still_tracks_the_vehicle_and_hides_the_ground_plane_and_the_series(recorded_flight, rrd_layout):
    """The Scene view still tracks the vehicle and hides the ground plane and the series.

    Given a recorded flight, when the run sends its default layout, then the Scene's eye tracks
    `/sim/model/shapes/shape_1` and its contents leave out `/sim/model/shapes/shape_0` and `/**/series/**`.
    """
    layout = rrd_layout(recorded_flight)

    assert (
        layout.view("Scene")["eye"],
        sorted(layout.hidden("Scene") & {"/sim/model/shapes/shape_0", "/**/series/**"}),
    ) == (
        "/sim/model/shapes/shape_1",
        ["/**/series/**", "/sim/model/shapes/shape_0"],
    )


def test_the_logs_view_shows_each_rows_entity_path_and_gathers_the_sims_log_rows_and_a_peers(
    recorded_flight, rrd_layout
):
    """The Logs view shows each row's entity path and gathers the sim's log rows and a peer's.

    Given a recorded flight, when the run sends its default layout, then the Logs view shows the
    EntityPath column, and its contents take in `/sim/logs/orchestrator` and `/px4_sitl/logs` and
    leave out `/sim/run/rtf`.
    """
    layout = rrd_layout(recorded_flight)
    shown = [e for e in ("/sim/logs/orchestrator", "/px4_sitl/logs", "/sim/run/rtf") if layout.shows("Logs", e)]

    assert (layout.view("Logs")["columns"].get(ENTITY_PATH_COLUMN), shown) == (
        True,
        ["/sim/logs/orchestrator", "/px4_sitl/logs"],
    )


def test_the_debug_tabs_show_one_tab_per_recorded_body_joint_and_sensor_rooted_at_the_instances_path(
    recorded_flight, rrd_layout
):
    """The debug tabs show one tab per recorded body, joint and sensor, rooted at the instance's path.

    Given a recorded flight with bodies, joints, an Inertial Measurement Unit (IMU) and a camera, when
    the run sends its default layout, then each instance has one tab, named as today, whose
    origin is its `/sim/vehicle/` path.
    """
    layout = rrd_layout(recorded_flight)
    tabs = ("body_frd", "rotor_1_ccw_joint", "imu (ImuSensor)", "fpvcam (RtxCameraSensor)")

    assert {tab: os.path.commonpath(layout.origins(tab)) for tab in tabs} == {
        "body_frd": "/sim/vehicle/body/body_frd/series",
        "rotor_1_ccw_joint": "/sim/vehicle/joints/rotor_1_ccw_joint/series",
        "imu (ImuSensor)": "/sim/vehicle/sensors/imu/series",
        "fpvcam (RtxCameraSensor)": "/sim/vehicle/sensors/fpvcam",
    }


def test_a_component_names_only_its_own_row_and_the_row_lands_under_the_components_path(recorded_flight, rrd_rows):
    """A component names only its own row, and the row lands under the component's path.

    Given a stand-in controller named `standin` that logs a strip as `horizon` through the logger the
    loop hands it, when a recorded flight ends, then the `.rrd` holds `/sim/vehicle/controllers/standin/horizon`.
    """
    strips = sorted({entity for entity, _, _ in rrd_rows(recorded_flight) if entity.endswith("/horizon")})

    assert strips == ["/sim/vehicle/controllers/standin/horizon"]
