"""Recorder: capturable taps write into device staging buffers that drain onto host blocks, the
read-side twin of logging. Exercises a body's history, the plant's `PoseTwist` signal tapped, a joint's
history, a struct of the joint's own widths, a sensor's history, tap → records of `t` then the signal's
quantities, and the drain past the staging buffer.
"""

import numpy as np
import pytest
import warp as wp

from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.core.schema import ImuSample, PoseTwist
from nexus_sim._src.core.signals import Signal, wire
from nexus_sim._src.core.stages import Bound
from nexus_sim._src.physics.outputs import joint_struct, write_body, write_joint
from nexus_sim._src.recording import History, Recorder

DT = 0.004


class _FakeView:
    """A one-body state stand-in: ``body_q`` = transform [p, q-xyzw], ``body_qd`` = spatial [lin, ang]."""

    def __init__(self, z):
        self.body_q = wp.array(np.array([[1.0, 2.0, z, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32), dtype=wp.transform)
        self.body_qd = wp.array(np.array([[0.0, 0.0, 0.5, 0.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.spatial_vector)


def _wired(*signals: Signal) -> tuple[Signal, ...]:
    """`signals`, each with the buffer a run's wiring gives an output: one stand-in stage writes them all."""
    stage = Stage("write", "device", lambda tick: None, writes=signals)
    wire([Bound(stage, object(), "sensor")])
    return signals


def _body_history(staging: int = 4096) -> tuple[Recorder, Signal, History]:
    """A Recorder with the base body's `PoseTwist` signal tapped under the plant's path, as a run taps it."""
    rec = Recorder(dt=DT, staging=staging)
    (body,) = _wired(Signal("body/body_frd", PoseTwist, shape=(1,)))
    return rec, body, rec.tap("vehicle", body, source="NewtonPhysics")


def _write(rec: Recorder, body: Signal, view: _FakeView) -> None:
    """Write `view` into the body's signal and record one row of every history, as a tick does."""
    wp.launch(write_body, dim=1, inputs=(view.body_q, view.body_qd, 0), outputs=(body.buffer,))
    rec.record()


def _heights(rows) -> list[float]:
    return [float(row["position"][2]) for row in rows]


def test_a_bodys_history_reads_as_records_of_t_then_the_pose_twists_fields():
    """A body's history, the plant's `PoseTwist` signal tapped, reads as records of `t` then `position`,
    `orientation`, `linear_velocity` and `angular_velocity`: `history()` every row, each one record.
    """
    rec, body, h = _body_history()
    for z in (1.0, 2.0, 3.0):
        _write(rec, body, _FakeView(z))
        rec.commit()
    latest = h.history()[-1]

    assert (
        h.history().dtype.names,
        latest["position"].tolist(),
        latest["orientation"].tolist(),
        latest["linear_velocity"].tolist(),
        _heights(h.history()),
        [round(float(t), 6) for t in h.history()["t"]],
    ) == (
        ("t", "position", "orientation", "linear_velocity", "angular_velocity"),
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.5],
        [1.0, 2.0, 3.0],
        [0.0, 0.004, 0.008],
    )


def test_a_joints_history_reads_as_records_of_q_and_qd_of_the_joints_own_widths():
    """A joint's history, its struct signal tapped, reads as records of `t`, `q` and `qd` of that joint's widths:
    a revolute joint's angle and rate from its slice of the flat arrays.
    """
    rec = Recorder(dt=DT)
    (joint,) = _wired(Signal("joints/rotor_1_joint", joint_struct(1, 1), shape=(1,)))
    h = rec.tap("vehicle", joint)
    jq = wp.array(np.array([9.9, 0.25], dtype=np.float32), dtype=float)  # the angle at index 1
    jqd = wp.array(np.array([9.9, 1.5], dtype=np.float32), dtype=float)
    wp.launch(write_joint, dim=1, inputs=(jq, jqd, 1, 1, 1, 1), outputs=(joint.buffer,))
    rec.record()
    rec.commit()
    row = h.history()[-1]

    assert (row.dtype.names, row["q"].tolist(), row["qd"].tolist()) == (("t", "q", "qd"), [0.25], [1.5])


def test_a_history_past_its_staging_buffer_holds_every_row_oldest_first():
    """A full staging buffer drains onto the host, so a history past it holds every row: 6 rows through a
    4-row buffer keep all 6, in order, and the newest row is still the last.
    """
    rec, body, h = _body_history(staging=4)
    for z in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        _write(rec, body, _FakeView(z))
        rec.commit()
    assert (_heights(h.history()), float(h.history()[-1]["position"][2])) == ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 6.0)


def test_a_commit_that_fills_the_staging_buffer_reports_the_drain():
    """`commit` says when it drained, every `staging` rows, and the device buffer keeps its size."""
    rec, body, h = _body_history(staging=3)
    drains = []
    for z in range(1, 8):
        _write(rec, body, _FakeView(float(z)))
        drains.append(h.commit())
    assert (drains, h.values.shape) == ([False, False, True, False, False, True, False], (3, 1))


def test_arrays_reads_a_row_range_across_blocks_and_the_staged_rows():
    """`arrays(start, stop)` reads one row range, whether it sits in the host blocks, on the device, or
    across the two, with the rows' own times.
    """
    rec, body, h = _body_history(staging=4)
    for z in range(1, 11):  # 10 rows: two blocks of 4 on the host, 2 still staged
        _write(rec, body, _FakeView(float(z)))
        rec.commit()
    across = h.arrays(3, 9)
    np.testing.assert_allclose(across["position"][:, 2], [4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    np.testing.assert_allclose(across["t"], np.arange(3, 9) * DT, atol=1e-9)
    assert (h.arrays(0, 4)["position"][:, 2].tolist(), h.arrays(8)["position"][:, 2].tolist()) == (
        [1.0, 2.0, 3.0, 4.0],
        [9.0, 10.0],
    )


def test_a_second_tap_of_one_key_raises():
    """A second tap of one key raises: one component writes a signal once, and two instances take two names."""
    rec, body, _ = _body_history()
    with pytest.raises(ValueError, match="already registered"):
        rec.tap("vehicle", body)


def test_the_recorder_commits_every_history_and_reports_the_drained_block():
    """One commit counts a row in every history; the commit that fills the staging buffers reports the
    block's row range, so the Logger writes those rows.
    """
    rec = Recorder(dt=DT, staging=2)
    body, joint = _wired(
        Signal("body/body_frd", PoseTwist, shape=(1,)), Signal("joints/j", joint_struct(1, 1), shape=(1,))
    )
    a, b = rec.tap("vehicle", body), rec.tap("vehicle", joint)
    blocks = []
    for z in (1.0, 2.0, 3.0, 4.0, 5.0):
        _write(rec, body, _FakeView(z))
        blocks.append(rec.commit())
    assert (blocks, len(a.history()), len(b.history())) == ([None, (0, 2), None, (2, 4), None], 5, 5)


def test_history_arrays_vectorizes_declared_fields():
    rec, body, h = _body_history()
    for z in (1.0, 2.0, 3.0):
        _write(rec, body, _FakeView(z))
        rec.commit()
    arrays = h.history_arrays()
    assert set(arrays) == {"t", "position", "orientation", "linear_velocity", "angular_velocity"}
    assert arrays["t"].shape == (3,)
    np.testing.assert_allclose(arrays["t"], [0.0, 0.004, 0.008], atol=1e-9)
    assert arrays["position"].shape == (3, 3)
    np.testing.assert_allclose(arrays["position"][:, 2], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(arrays["orientation"][-1], [0.0, 0.0, 0.0, 1.0])
    assert arrays["linear_velocity"].shape == (3, 3)


def test_history_arrays_spans_the_drained_blocks_and_handles_empty():
    rec, body, h = _body_history(staging=4)
    assert h.history_arrays()["t"].shape == (0,)  # before the first tick: N == 0, no raise
    for z in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):  # 6 rows through a 4-row staging buffer: one block drained
        _write(rec, body, _FakeView(z))
        rec.commit()
    arrays = h.history_arrays()
    np.testing.assert_allclose(arrays["position"][:, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # oldest first, every row
    assert np.all(np.diff(arrays["t"]) > 0)  # counter × dt stays monotonic across the drain


def test_fields_schema_must_cover_width():
    with pytest.raises(ValueError, match="width"):
        History(width=13, dt=DT, fields=(("position", 3),))


def test_the_recorder_taps_a_signal_under_its_writers_path_with_the_quantities_its_type_declares():
    """`tap` registers a signal's history as `<path>/<signal>`, its quantities those of the signal's type: a
    struct's fields, each with its width, or one quantity named after a value-typed signal, of its width.
    """
    rec = Recorder(dt=DT)
    imu, controls = _wired(Signal("imu", ImuSample, shape=(1,)), Signal("controls", wp.float32, shape=(1, 16)))
    a = rec.tap("vehicle/sensors/imu", imu, source="ImuSensor")
    b = rec.tap("vehicle/controllers/px4", controls, source="Px4Controller")

    assert (sorted(rec.histories), a.fields, b.fields, a.source) == (
        ["vehicle/controllers/px4/controls", "vehicle/sensors/imu/imu"],
        (("time", 1), ("accel", 3), ("gyro", 3)),
        (("controls", 16),),
        "ImuSensor",
    )


def test_the_record_stage_copies_each_tapped_signals_value_into_a_row_of_t_then_its_quantities():
    """Each record copies what every tapped signal holds into its history, and the history reads as records of
    `t` then the signal's quantities, with the struct's own types, across a drain.
    """
    rec = Recorder(dt=DT, staging=2)
    imu, controls = _wired(Signal("imu", ImuSample, shape=(1,)), Signal("controls", wp.float32, shape=(1, 16)))
    rec.tap("vehicle/sensors/imu", imu)
    rec.tap("vehicle/controllers/px4", controls)
    for k in (1, 2, 3):
        imu.write([(0.5 * k, (k, 0.0, 0.0), (0.0, k, 0.0))])
        controls.write(np.full((1, 16), k, dtype=np.float32))
        rec.record()
        rec.commit()
    rows = rec.histories["vehicle/sensors/imu/imu"].history()
    newest = rec.histories["vehicle/controllers/px4/controls"].history()[-1]

    assert (
        rows["t"].tolist(),
        rows["time"].tolist(),
        rows["accel"][:, 0].tolist(),
        str(rows["time"].dtype),
        str(rows["accel"].dtype),
        float(newest["t"]),
        newest["controls"].tolist(),
    ) == (
        pytest.approx([0.0, 0.004, 0.008]),
        [0.5, 1.0, 1.5],
        [1.0, 2.0, 3.0],
        "float64",
        "float32",
        pytest.approx(0.008),
        [3.0] * 16,
    )
