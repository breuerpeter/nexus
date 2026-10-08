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
