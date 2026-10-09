"""Recorder: capturable taps write into device staging buffers that drain onto host blocks, the
read-side twin of logging. Exercises a body's history, record_body → BodyState, a joint's history,
record_joint → JointState with variable width, the drain past the staging buffer, and the Histories
name view.
"""

import numpy as np
import pytest
import warp as wp

from nexus_sim._src.recording import BodyState, Histories, History, JointState, Recorder
from nexus_sim._src.recording.state import (
    BODY_FIELDS,
    BODY_WIDTH,
    decode_body,
    make_decode_joint,
    record_body,
    record_joint,
)


class _FakeView:
    """A one-body state stand-in: ``body_q`` = transform [p, q-xyzw], ``body_qd`` = spatial [lin, ang]."""

    def __init__(self, z):
        self.body_q = wp.array(np.array([[1.0, 2.0, z, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32), dtype=wp.transform)
        self.body_qd = wp.array(np.array([[0.0, 0.0, 0.5, 0.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.spatial_vector)


def _rec_body(ch: History, view: _FakeView) -> None:
    """Record one row of `view` into `ch`, as the loop does: the kernel, then the host's commit."""
    wp.launch(record_body, dim=1, inputs=(view.body_q, view.body_qd, 0, ch.dt, ch.staging, ch.buf, ch.counter))
    ch.commit()


def test_body_history_records_latest_and_history():
    ch = History(width=BODY_WIDTH, dt=0.004, decode=decode_body)
    for z in (1.0, 2.0, 3.0):
        _rec_body(ch, _FakeView(z))

    latest = ch.latest()
    assert isinstance(latest, BodyState)
    assert latest.position == (1.0, 2.0, 3.0)
    assert latest.quat_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert latest.velocity == (0.0, 0.0, 0.5)
    assert latest.altitude_m == 3.0  # world Z-up, not -z

    hist = ch.history()
    assert [s.altitude_m for s in hist] == [1.0, 2.0, 3.0]
    assert [round(s.t, 6) for s in hist] == [0.0, 0.004, 0.008]  # counter × dt, capture-correct


def test_joint_history_records_variable_width():
    # A revolute joint: nq=1, the angle, nqd=1, the rate. width = 1(t) + 1 + 1.
    ch = History(width=3, dt=0.004, decode=make_decode_joint(1, 1))
    jq = wp.array(np.array([0.25, 9.9], dtype=np.float32), dtype=float)  # angle at index 0; 9.9 = other dof
    jqd = wp.array(np.array([1.5, 9.9], dtype=np.float32), dtype=float)
    wp.launch(record_joint, dim=1, inputs=(jq, jqd, 0, 1, 0, 1, ch.dt, ch.staging, ch.buf, ch.counter))
    js = ch.latest()
    assert isinstance(js, JointState)
    assert js.q == (0.25,)
    assert js.qd == (1.5,)


def test_a_history_past_its_staging_buffer_holds_every_row_oldest_first():
    """A full staging buffer drains onto the host, so a history past it holds every row: 6 rows through a
    4-row buffer keep all 6, in order, and the newest row is still the last.
    """
    ch = History(width=BODY_WIDTH, dt=0.004, decode=decode_body, staging=4)
    for z in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        _rec_body(ch, _FakeView(z))
    assert ([s.altitude_m for s in ch.history()], ch.latest().altitude_m) == ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 6.0)


def test_a_commit_that_fills_the_staging_buffer_reports_the_drain():
    """`commit` says when it drained, every `staging` rows, and the device buffer keeps its size."""
    ch = History(width=BODY_WIDTH, dt=0.004, decode=decode_body, staging=3)
    drains = []
    for z in range(1, 8):
        wp.launch(
            record_body,
            dim=1,
            inputs=(_FakeView(float(z)).body_q, _FakeView(float(z)).body_qd, 0, ch.dt, ch.staging, ch.buf, ch.counter),
        )
        drains.append(ch.commit())
    assert (drains, ch.buf.shape) == ([False, False, True, False, False, True, False], (3, BODY_WIDTH))


def test_arrays_reads_a_row_range_across_blocks_and_the_staged_rows():
    """`arrays(start, stop)` reads one row range, whether it sits in the host blocks, on the device, or
    across the two, with the rows' own times.
    """
    ch = History(width=BODY_WIDTH, dt=0.004, decode=decode_body, staging=4, fields=BODY_FIELDS)
    for z in range(1, 11):  # 10 rows: two blocks of 4 on the host, 2 still staged
        _rec_body(ch, _FakeView(float(z)))
    across = ch.arrays(3, 9)
    np.testing.assert_allclose(across["position"][:, 2], [4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    np.testing.assert_allclose(across["t"], np.arange(3, 9) * 0.004, atol=1e-9)
    assert (ch.arrays(0, 4)["position"][:, 2].tolist(), ch.arrays(8)["position"][:, 2].tolist()) == (
        [1.0, 2.0, 3.0, 4.0],
        [9.0, 10.0],
    )


def test_history_latest_before_any_row_raises():
    with pytest.raises(RuntimeError):
        History(width=BODY_WIDTH, dt=0.004, decode=decode_body).latest()


def test_recorder_registers_named_histories_idempotently():
    rec = Recorder(dt=0.004)
    a = rec.history("vehicle/body/body_frd", width=BODY_WIDTH, decode=decode_body)
    b = rec.history("vehicle/body/body_frd", width=BODY_WIDTH, decode=decode_body)
    assert a is b  # idempotent registration → one buffer per name
    assert set(rec.histories) == {"vehicle/body/body_frd"}


def test_the_recorder_commits_every_history_and_reports_the_drained_block():
    """One commit counts a row in every history; the commit that fills the staging buffers reports the
    block's row range, so the Logger writes those rows.
    """
    rec = Recorder(dt=0.004, staging=2)
    a = rec.history("vehicle/body/body_frd", width=BODY_WIDTH, decode=decode_body)
    b = rec.history("vehicle/joints/rotor_1_joint", width=3, decode=make_decode_joint(1, 1))
    blocks = []
    for z in (1.0, 2.0, 3.0, 4.0, 5.0):
        for ch in (a, b):
            wp.launch(
                record_body,
                dim=1,
                inputs=(_FakeView(z).body_q, _FakeView(z).body_qd, 0, ch.dt, ch.staging, ch.buf, ch.counter),
            ) if ch is a else None
        jq = wp.array(np.array([z, 0.0], dtype=np.float32), dtype=float)
        wp.launch(record_joint, dim=1, inputs=(jq, jq, 0, 1, 0, 1, b.dt, b.staging, b.buf, b.counter))
        blocks.append(rec.commit())
    assert (blocks, len(a.history()), len(b.history())) == ([None, (0, 2), None, (2, 4), None], 5, 5)


def test_history_arrays_vectorizes_declared_fields():
    ch = History(width=BODY_WIDTH, dt=0.004, decode=decode_body, fields=BODY_FIELDS)
    for z in (1.0, 2.0, 3.0):
        _rec_body(ch, _FakeView(z))
    arrays = ch.history_arrays()
    assert set(arrays) == {"t", "position", "quat_xyzw", "velocity", "angular_velocity"}
    assert arrays["t"].shape == (3,)
    np.testing.assert_allclose(arrays["t"], [0.0, 0.004, 0.008], atol=1e-9)
    assert arrays["position"].shape == (3, 3)
    np.testing.assert_allclose(arrays["position"][:, 2], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(arrays["quat_xyzw"][-1], [0.0, 0.0, 0.0, 1.0])
    assert arrays["velocity"].shape == (3, 3)


def test_history_arrays_spans_the_drained_blocks_and_handles_empty():
    ch = History(width=BODY_WIDTH, dt=0.004, decode=decode_body, staging=4, fields=BODY_FIELDS)
    assert ch.history_arrays()["t"].shape == (0,)  # before the first tick: N == 0, no raise
    for z in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):  # 6 rows through a 4-row staging buffer: one block drained
        _rec_body(ch, _FakeView(z))
    arrays = ch.history_arrays()
    np.testing.assert_allclose(arrays["position"][:, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # oldest first, every row
    assert np.all(np.diff(arrays["t"]) > 0)  # counter × dt stays monotonic across the drain


def test_history_arrays_requires_declared_fields():
    ch = History(width=BODY_WIDTH, dt=0.004, decode=decode_body)  # no schema
    with pytest.raises(ValueError, match="fields"):
        ch.history_arrays()


def test_fields_schema_must_cover_width():
    with pytest.raises(ValueError, match="width"):
        History(width=BODY_WIDTH, dt=0.004, decode=decode_body, fields=(("position", 3),))


def test_duplicate_name_with_different_shape_raises():
    rec = Recorder(dt=0.004)
    rec.history("vehicle/sensors/imu", width=11, decode=lambda r: r, source="ImuBosch")
    with pytest.raises(ValueError, match="distinct instance names"):
        rec.history("vehicle/sensors/imu", width=11, decode=lambda r: r, source="ImuMurata")
    with pytest.raises(ValueError, match="distinct instance names"):
        rec.history("vehicle/sensors/imu", width=4, decode=lambda r: r, source="ImuBosch")


def test_histories_resolve_bare_names_across_namespaces():
    rec = Recorder(dt=0.004)
    rec.history("vehicle/body/body_frd", width=BODY_WIDTH, decode=decode_body)
    rec.history("vehicle/joints/rotor_1_joint", width=3, decode=make_decode_joint(1, 1))
    rec.history("vehicle/sensors/imu", width=11, decode=lambda r: r)
    state = Histories(rec.histories, ("vehicle/body/", "vehicle/joints/"))
    meas = Histories(rec.histories, ("vehicle/sensors/",))

    assert rec.histories["vehicle/body/body_frd"] is state["body_frd"]
    assert rec.histories["vehicle/joints/rotor_1_joint"] is state["rotor_1_joint"]
    assert set(state) == {"body_frd", "rotor_1_joint"}  # spans both prefixes; sensors excluded
    assert set(meas) == {"imu"}
    with pytest.raises(KeyError):
        state["imu"]  # not in body/ or joint/
