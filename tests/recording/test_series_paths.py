"""Where the Recorder's series and the flown path sit in a run's recording.

The tests read the session's one recorded flight of the shipped camera vehicle, which
``tests/conftest.py`` flies on the Warp CPU backend with a Recorder attached. Skipped if rerun or
newton are missing.
"""

import pytest

pytest.importorskip("rerun")
pytest.importorskip("newton")


def test_each_recorded_series_sits_under_its_instance_in_a_series_child(recorded_flight, rrd_rows):
    """Each recorded series sits under its instance, in a `series` child.

    Given a recorded flight of a vehicle with rotor joints and an Inertial Measurement Unit (IMU), the
    recorder on, when it ends, then the `.rrd` holds each declared field of the base body at
    `/sim/vehicle/body/body_frd/series/<field>`, of a rotor joint at
    `/sim/vehicle/joints/rotor_1_ccw_joint/series/<field>` and of the IMU at
    `/sim/vehicle/sensors/imu/series/<field>`, and no entity under `/recording/`.
    """
    rows = rrd_rows(recorded_flight)
    fields: dict[str, set[str]] = {}
    for entity, _, columns in rows:
        if "/series/" in entity and "Scalars:scalars" in columns:
            instance, field = entity.split("/series/")
            fields.setdefault(instance, set()).add(field)
    instances = ("/sim/vehicle/body/body_frd", "/sim/vehicle/joints/rotor_1_ccw_joint", "/sim/vehicle/sensors/imu")
    old = sorted({entity for entity, _, _ in rows if entity.startswith("/recording/")})

    assert ({instance: sorted(fields.get(instance, ())) for instance in instances}, old) == (
        {
            "/sim/vehicle/body/body_frd": ["angular_velocity", "position", "quat_xyzw", "velocity"],
            "/sim/vehicle/joints/rotor_1_ccw_joint": ["q", "qd"],
            "/sim/vehicle/sensors/imu": ["xacc", "xgyro", "yacc", "ygyro", "zacc", "zgyro"],
        },
        [],
    )


def test_the_flown_path_sits_at_sim_vehicle_trajectory(recorded_flight, rrd_rows):
    """The flown path sits at `sim/vehicle/trajectory`.

    Given a recorded flight with the recorder on, when it ends, then the `.rrd` holds a strip at
    `/sim/vehicle/trajectory` and no entity under `/physics/`.
    """
    rows = rrd_rows(recorded_flight)
    strips = sorted({e for e, _, columns in rows if e.endswith("/trajectory") and "LineStrips3D:strips" in columns})
    old = sorted({entity for entity, _, _ in rows if entity.startswith("/physics/")})

    assert (strips, old) == (["/sim/vehicle/trajectory"], [])


def _origins(layout, tab: str) -> list[str]:
    """The origin of every view under the tab named `tab`, or none when the layout has no such tab."""
    try:
        return layout.origins(tab)
    except ValueError:
        return []


def test_each_quantity_of_a_history_is_one_series_at_sim_key_field_with_no_series_level(
    recorded_flight, rrd_rows, rrd_layout
):
    """The recording holds each quantity of a history as one series at `sim/<key>/<field>`, with no `series`
    level, and the viewer's debug tabs group them by role as the sensors' are today.

    Given the session's recorded flight of the camera vehicle with the IMU, the passthrough estimator and the
    stand-in controller, when it ends, then the `.rrd` holds `time`, `accel` and `gyro` at
    `sim/vehicle/sensors/imu/imu/<field>`, the estimate's quantities at
    `sim/vehicle/estimators/<name>/estimate/<field>`, `controls` at
    `sim/vehicle/controllers/standin/controls/controls`, the base body's `position` at
    `sim/vehicle/body/body_frd/position`, each with one row per tick, and no entity with a `series` segment.
    The estimator's instance is its class's name until #206 names it from its prim.
    """
    from tests.conftest import FLIGHT_TICKS

    held: dict[str, int] = {}
    for entity, _, columns in rrd_rows(recorded_flight):
        if "Scalars:scalars" in columns:
            held[entity] = held.get(entity, 0) + len(columns["Scalars:scalars"])
    imu = "/sim/vehicle/sensors/imu/imu"
    estimate = "/sim/vehicle/estimators/GroundTruthEstimator/estimate"
    controls = "/sim/vehicle/controllers/standin/controls/controls"
    wanted = (
        f"{imu}/time",
        f"{imu}/accel",
        f"{imu}/gyro",
        f"{estimate}/position",
        controls,
        "/sim/vehicle/body/body_frd/position",
    )
    layout = rrd_layout(recorded_flight)
    grouped = {
        tab: entity in _origins(layout, tab)
        for tab, entity in (
            ("Sensors", f"{imu}/accel"),
            ("Estimators", f"{estimate}/position"),
            ("Controllers", controls),
        )
    }

    assert ({e: held.get(e, 0) for e in wanted}, sorted(e for e in held if "/series/" in e), grouped) == (
        dict.fromkeys(wanted, FLIGHT_TICKS + 1),
        [],
        {"Sensors": True, "Estimators": True, "Controllers": True},
    )
