"""newton-logging: the central Rerun recording, Logger: scene + events into one .rrd.

Real end-to-end in file mode, serve=False, on the Warp CPU backend; skipped if rerun/newton/pxr
are missing. File mode keeps the tests hermetic: no gRPC server/port, just an inspectable .rrd.
"""

import pytest

pytest.importorskip("rerun")
pytest.importorskip("newton")


def _rrd_entities(path: str) -> list[str]:
    # rerun 0.34 dropped the local dataframe API, ``rerun.recording.load_recording``; reading an
    # rrd now needs the catalog server + the optional datafusion dep. The bundled, version-matched
    # command-line tool still prints per-chunk entity paths, so parse those instead.
    import re
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "rerun", "rrd", "print", path], capture_output=True, text=True, check=True
    ).stdout
    return sorted(set(re.findall(r" - (/\S*) - data columns:", out)))


def test_recording_path_under_cache():
    """``recording_path`` keeps named recordings out of the working tree, in the newton cache."""
    from nexus._src.logging import recording_path

    p = recording_path("demo")
    assert "recordings" in p and p.endswith("demo.rrd") and "/.cache/nexus/" in p


def test_scene_only_blueprint_has_no_logs_view():
    """The examples' blueprint shows only the 3D scene: no ``logs`` TextLog view."""
    import rerun.blueprint as rrb

    from nexus._src.logging import scene_only_blueprint

    bp = scene_only_blueprint()
    assert isinstance(bp, rrb.Blueprint)
    # the standard layout references the "logs" entity in a TextLogView; the scene-only one must not
    assert "logs" not in repr(bp)


def test_single_blueprint_owned_by_rerunlogger(tmp_path):
    """Logger owns the blueprint. ViewerRerun's ctor would otherwise log its own ``origin="/"``
    blueprint as the rrd's ``default_blueprint``, on top of Logger's own, leaving two competing
    blueprints, and a viewer honouring the default re-introduces panels/entities the examples scoped
    out. Logger strips that, so the recording carries exactly one blueprint view.
    """
    import pathlib
    import re

    from nexus._src.logging import Logger, scene_only_blueprint

    rrd = str(tmp_path / "one_bp.rrd")
    Logger(model=None, serve=False, record_to_rrd=rrd, blueprint=scene_only_blueprint()).close()
    view_ids = set(re.findall(rb"/view/[0-9a-f-]{36}", pathlib.Path(rrd).read_bytes()))
    assert len(view_ids) == 1, f"expected one blueprint view, found {len(view_ids)}: {view_ids}"


def test_custom_blueprint_accepted(tmp_path):
    """``Logger(blueprint=...)`` plumbs through and still produces a valid recording."""
    from nexus._src.logging import Logger, scene_only_blueprint

    rrd = str(tmp_path / "bp.rrd")
    rl = Logger(model=None, serve=False, record_to_rrd=rrd, blueprint=scene_only_blueprint())
    from nexus._src.core import logger

    logger.info("event with a custom blueprint")
    rl.close()

    assert any("logs/sim" in p for p in _rrd_entities(rrd))


def test_events_routed_to_logs_sim(tmp_path):
    """Logger tees the ``newton`` logger into the recording's ``logs/sim`` panel: a
    component just calls ``logger.info(...)`` and it lands in the same recording.
    """
    from nexus._src.core import logger
    from nexus._src.logging import Logger

    rrd = str(tmp_path / "events.rrd")
    rl = Logger(model=None, serve=False, record_to_rrd=rrd)
    logger.info("hello from a component")
    rl.close()

    paths = _rrd_entities(rrd)
    assert any("logs/sim" in p for p in paths), paths


def test_blueprint_layout_and_eye_tracking():
    """The default layout is two rows: the top row's two equal columns are [the one-line
    Real Time Factor (RTF) readout over the eye-tracked Scene] and [the Logs|Settings tab container];
    the bottom row is the debug tab tree at *full* viewer width. Cameras are ordinary Sensors instance
    tabs, no special panel.
    """
    from nexus._src.logging import FPV_ENTITY
    from nexus._src.logging.rerun_logging import VEHICLE_SHAPE_ENTITY, _blueprint

    bp = _blueprint(cameras={"fpvcam": "RtxCameraSensor", "lr1cam": None}, has_settings=True)
    rows = bp.root_container.contents
    assert len(rows) == 2  # [RTF+Scene | Logs|Settings] over the full-width debug tabs
    assert list(bp.root_container.row_shares) == [1.0, 1.0]
    top, bottom = rows
    left, log_tabs = top.contents
    assert list(top.column_shares) == [1.0, 1.0]
    assert [getattr(v, "name", None) for v in log_tabs.contents] == ["Logs", "Settings"]
    rtf, scene = left.contents
    assert getattr(rtf, "name", None) == "RTF"
    assert list(left.row_shares) == [1.0, 8.0]  # ≈ one line of markdown over the scene
    assert getattr(scene, "name", None) == "Scene"
    # the 3D eye tracks the vehicle mesh
    eye = scene.properties["EyeControls3D"]
    assert VEHICLE_SHAPE_ENTITY in str(eye.tracking_entity.as_arrow_array().to_pylist())
    (sensors_tab,) = bottom.contents  # cameras land as instance tabs under Sensors
    assert getattr(sensors_tab, "name", None) == "Sensors"
    assert [getattr(v, "name", None) for v in sensors_tab.contents] == ["fpvcam (RtxCameraSensor)", "lr1cam"]
    assert [str(getattr(v, "origin", None)) for v in sensors_tab.contents] == [
        f"{FPV_ENTITY}/fpvcam",
        f"{FPV_ENTITY}/lr1cam",
    ]

    # no cameras/recording → the First Person View (FPV) placeholder in the bottom row; no settings → no Settings tab
    bp0 = _blueprint()
    top0, bottom0 = bp0.root_container.contents
    assert getattr(bottom0, "name", None) == "FPV"
    assert [getattr(v, "name", None) for v in top0.contents[1].contents] == ["Logs"]


def test_blueprint_recording_tabs_mirror_channel_keys():
    """The debug tab tree derives from the Recorder's channel keys: group ▸ instance ▸ quantity
    mirrors the access surface, ``sim.physics["body_frd"]`` → Physics ▸ body_frd ▸ position …, with
    sensor instance tabs carrying the impl class.
    """
    from nexus._src.logging.rerun_logging import RECORDING_ROOT, _blueprint

    recording = {
        "physics/body/body_frd": ("NewtonPhysics", ["position", "velocity"]),
        "physics/joint/rotor_1_joint": ("NewtonPhysics", ["q", "qd"]),
        "sensors/imu": ("ImuSensor", ["xacc", "ygyro"]),
    }
    bp = _blueprint(recording=recording, cameras={"fpvcam": "RtxCameraSensor"})
    _top, bottom = bp.root_container.contents  # the debug tabs are the full-width bottom row
    groups = {getattr(g, "name", None): g for g in bottom.contents}
    assert list(groups) == ["Physics", "Sensors"]
    body = groups["Physics"].contents[0]
    assert getattr(body, "name", None) == "body_frd"  # instance tab = the sim.physics[...] key
    assert [v.name for v in body.contents] == ["position", "velocity"]  # quantity tabs = the fields
    assert [str(v.origin) for v in body.contents] == [
        f"{RECORDING_ROOT}/physics/body/body_frd/position",
        f"{RECORDING_ROOT}/physics/body/body_frd/velocity",
    ]
    assert getattr(groups["Physics"].contents[1], "name", None) == "rotor_1_joint"
    # scalar sensors and the camera feed are SIBLING instance tabs under Sensors
    assert [getattr(v, "name", None) for v in groups["Sensors"].contents] == [
        "imu (ImuSensor)",
        "fpvcam (RtxCameraSensor)",
    ]


def test_settings_markdown_is_one_flat_dotted_table():
    """The Settings tab renders one Setting|Value table, and the Profile tab reuses the renderer with a
    "Property" header: every leaf keyed by its fully dotted
    config path; scalar lists inline comma-separated; empty sections stay visible.
    """
    from nexus._src.logging.rerun_logging import _settings_markdown

    md = _settings_markdown(
        {
            "scenario": {"physics": {"dt": 0.004, "solver": "mujoco"}, "seed": 42},
            "example": {"goal_w": [3.0, 0.0, 2.0], "mods": [{"key": "esc", "value": 2}]},
            "sensors": {},
        }
    )
    lines = md.splitlines()
    assert lines[0] == "| Setting | Value |" and lines[1] == "|---|---|"
    assert _settings_markdown({"rtf": 1.2}, key_header="Property").startswith("| Property | Value |")  # Profile tab
    assert "| scenario.physics.dt | 0.004 |" in lines  # nesting rides the dotted path
    assert "| scenario.physics.solver | mujoco |" in lines
    assert "| scenario.seed | 42 |" in lines
    assert "| example.goal_w | 3.0, 0.0, 2.0 |" in lines  # scalar list inline, comma-separated
    assert "| example.mods | key: esc, value: 2 |" in lines  # list of mappings, entries inlined
    assert "| sensors | (empty) |" in lines  # empty section stays visible
    assert all(line.startswith("|") for line in lines)  # nothing outside the one table


def test_settings_and_rtf_land_in_recording(tmp_path):
    """``settings=`` logs the one static run-settings doc; ``log_rtf`` logs the live readout, both
    under the ``run/`` namespace the right column's views read.
    """
    from nexus._src.logging import Logger

    rrd = str(tmp_path / "run_docs.rrd")
    rl = Logger(model=None, serve=False, record_to_rrd=rrd, settings={"runtime": {"dt": 0.004}, "seed": 42})
    rl.set_time(0.5)
    rl.log_rtf(1.23)
    rl.log_profile({"control_steps": 100, "rtf": 1.2, "profile": {"tick_p50_ms": 0.04}})
    assert rl._has_profile  # the rebuilt blueprint now carries the Profile tab
    rl.close()
    paths = _rrd_entities(rrd)
    assert any("run/settings" in p for p in paths), paths
    assert any("run/rtf" in p for p in paths), paths
    assert any("run/profile" in p for p in paths), paths


def test_default_blueprint_with_fpv_accepted(tmp_path):
    """A recording built with the default blueprint, with its FPV column, comes out without error."""
    import pathlib

    from nexus._src.logging import Logger

    rrd = str(tmp_path / "fpv_bp.rrd")
    Logger(model=None, serve=False, record_to_rrd=rrd).close()  # default _blueprint now has FPV col
    assert pathlib.Path(rrd).exists()


def test_log_image_lands_in_recording(tmp_path):
    """``log_image`` writes the frame to its entity, ``cameras/fpv``, in the recording."""
    import numpy as np

    from nexus._src.logging import FPV_ENTITY, Logger

    rrd = str(tmp_path / "img.rrd")
    rl = Logger(model=None, serve=False, record_to_rrd=rrd)
    rl.log_image(FPV_ENTITY, np.zeros((8, 8, 3), dtype=np.uint8))
    rl.close()
    assert any(FPV_ENTITY in p for p in _rrd_entities(rrd))


def test_log_image_fault_isolated_and_warns_once(tmp_path, monkeypatch):
    """A broken ``rr.log`` must *not* raise out of ``log_image``, which is output-only, and warns exactly
    once, since the encode worker is fault-isolated; this mirrors the scene ``log`` fault-isolation.
    """
    import numpy as np
    import rerun as rr

    from nexus._src.logging import FPV_ENTITY, Logger

    rl = Logger(model=None, serve=False, record_to_rrd=str(tmp_path / "broken.rrd"))

    calls = {"n": 0}

    def boom(entity, *a, **k):
        # Only break the FPV image path; the event-log handler also routes via rr.log, to logs/sim, and
        # must keep working so close() doesn't trip on it.
        if entity == FPV_ENTITY:
            calls["n"] += 1
            raise RuntimeError("rr.log is broken")

    monkeypatch.setattr(rr, "log", boom)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rl.log_image(FPV_ENTITY, rgb)  # must not raise
    rl.log_image(FPV_ENTITY, rgb)  # must not raise
    rl.close()

    assert calls["n"] == 1  # the worker stops after the first failure: warn once, feed off
    assert rl._img_log_warned is True  # warned, and the flag latches it to once


def test_log_image_no_logger_side_throttle(tmp_path, monkeypatch):
    """The producing *sensor* owns the frame rate, by sim-time decimation; the Logger must log every frame
    handed to it, since a second, logger-side throttle halved the effective fps.
    """
    import numpy as np
    import rerun as rr

    from nexus._src.logging import FPV_ENTITY, Logger

    rl = Logger(model=None, serve=False, record_to_rrd=str(tmp_path / "dec.rrd"))
    n = {"c": 0}

    def count(entity, *a, **k):
        if entity == FPV_ENTITY:  # ignore the event-log handler's own rr.log calls to logs/sim
            n["c"] += 1

    monkeypatch.setattr(rr, "log", count)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rl.log_image(FPV_ENTITY, rgb)
    rl.log_image(FPV_ENTITY, rgb)
    rl.close()  # drains the encode worker
    assert n["c"] == 2  # both frames logged: no logger-side throttle


@pytest.mark.usefixtures("warp_cpu")
def test_scene_logged_via_log_state(tmp_path):
    """Logger.log(t, state) drives NVIDIA Newton's ViewerRerun.log_state, so the vehicle
    geometry + its evolving pose land in the recording.
    """
    pytest.importorskip("pxr")
    import newton
    from pxr import Usd, UsdGeom, UsdPhysics

    from nexus._src.logging import Logger
    from nexus._src.physics.builders.usd import USDBuilder

    usd = str(tmp_path / "mini.usda")
    stage = Usd.Stage.CreateNew(usd)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    body = UsdGeom.Cube.Define(stage, "/Vehicle/body")
    body.GetSizeAttr().Set(0.2)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).GetMassAttr().Set(1.5)
    stage.GetRootLayer().Save()

    mb = newton.ModelBuilder()
    USDBuilder({"usd_path": usd}, None).build(mb)
    model = mb.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    rrd = str(tmp_path / "scene.rrd")
    rl = Logger(model, serve=False, record_to_rrd=rrd)
    for i in range(3):
        rl.log_state(state, i * 0.01)
    rl.close()

    paths = _rrd_entities(rrd)
    assert any(p.startswith("/model") or p.startswith("/geometry") for p in paths), paths
