"""Recorder seam: capturable per-entity taps write into device ring buffers the host reads on demand,
the read-side twin of logging. Exercises the body channel, record_body → BodyState, the joint channel,
record_joint → JointState with variable width, and the ChannelMap name view.
"""

import numpy as np
import pytest
import warp as wp

from nexus._src.recording import BodyState, ChannelMap, JointState, RecordChannel, Recorder
from nexus._src.recording.state import (
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


def _rec_body(ch: RecordChannel, view: _FakeView) -> None:
    wp.launch(record_body, dim=1, inputs=(view.body_q, view.body_qd, 0, ch.dt, ch.maxlen, ch.buf, ch.counter))


def test_body_channel_records_latest_and_history():
    ch = RecordChannel(width=BODY_WIDTH, dt=0.004, decode=decode_body)
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


def test_joint_channel_records_variable_width():
    # A revolute joint: nq=1, the angle, nqd=1, the rate. width = 1(t) + 1 + 1.
    ch = RecordChannel(width=3, dt=0.004, decode=make_decode_joint(1, 1))
    jq = wp.array(np.array([0.25, 9.9], dtype=np.float32), dtype=float)  # angle at index 0; 9.9 = other dof
    jqd = wp.array(np.array([1.5, 9.9], dtype=np.float32), dtype=float)
    wp.launch(record_joint, dim=1, inputs=(jq, jqd, 0, 1, 0, 1, ch.dt, ch.maxlen, ch.buf, ch.counter))
    js = ch.latest()
    assert isinstance(js, JointState)
    assert js.q == (0.25,)
    assert js.qd == (1.5,)


def test_channel_ring_wraps_oldest_first():
    ch = RecordChannel(width=BODY_WIDTH, dt=0.004, decode=decode_body, maxlen=4)
    for z in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):  # 6 rows into a 4-slot ring → oldest two dropped
        _rec_body(ch, _FakeView(z))
    assert [s.altitude_m for s in ch.history()] == [3.0, 4.0, 5.0, 6.0]  # oldest-first, wrapped
    assert ch.latest().altitude_m == 6.0


def test_channel_latest_before_any_row_raises():
    with pytest.raises(RuntimeError):
        RecordChannel(width=BODY_WIDTH, dt=0.004, decode=decode_body).latest()


def test_recorder_registers_named_channels_idempotently():
    rec = Recorder(dt=0.004)
    a = rec.channel("physics/body/body_frd", width=BODY_WIDTH, decode=decode_body)
    b = rec.channel("physics/body/body_frd", width=BODY_WIDTH, decode=decode_body)
    assert a is b  # idempotent registration → one buffer per name
    assert set(rec.channels) == {"physics/body/body_frd"}


def test_history_arrays_vectorizes_declared_fields():
    ch = RecordChannel(width=BODY_WIDTH, dt=0.004, decode=decode_body, fields=BODY_FIELDS)
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


def test_history_arrays_unwraps_ring_and_handles_empty():
    ch = RecordChannel(width=BODY_WIDTH, dt=0.004, decode=decode_body, maxlen=4, fields=BODY_FIELDS)
    assert ch.history_arrays()["t"].shape == (0,)  # before the first tick: N == 0, no raise
    for z in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):  # 6 rows into a 4-slot ring → oldest two dropped
        _rec_body(ch, _FakeView(z))
    arrays = ch.history_arrays()
    np.testing.assert_allclose(arrays["position"][:, 2], [3.0, 4.0, 5.0, 6.0])  # oldest-first, wrapped
    assert np.all(np.diff(arrays["t"]) > 0)  # counter × dt stays monotonic across the wrap


def test_history_arrays_requires_declared_fields():
    ch = RecordChannel(width=BODY_WIDTH, dt=0.004, decode=decode_body)  # no schema
    with pytest.raises(ValueError, match="fields"):
        ch.history_arrays()


def test_fields_schema_must_cover_width():
    with pytest.raises(ValueError, match="width"):
        RecordChannel(width=BODY_WIDTH, dt=0.004, decode=decode_body, fields=(("position", 3),))


def test_duplicate_name_with_different_shape_raises():
    rec = Recorder(dt=0.004)
    rec.channel("sensors/imu", width=11, decode=lambda r: r, source="ImuBosch")
    with pytest.raises(ValueError, match="distinct instance names"):
        rec.channel("sensors/imu", width=11, decode=lambda r: r, source="ImuMurata")
    with pytest.raises(ValueError, match="distinct instance names"):
        rec.channel("sensors/imu", width=4, decode=lambda r: r, source="ImuBosch")


def test_channel_map_resolves_bare_names_across_namespaces():
    rec = Recorder(dt=0.004)
    rec.channel("physics/body/body_frd", width=BODY_WIDTH, decode=decode_body)
    rec.channel("physics/joint/rotor_1_joint", width=3, decode=make_decode_joint(1, 1))
    rec.channel("sensors/imu", width=11, decode=lambda r: r)
    state = ChannelMap(rec.channels, ("physics/body/", "physics/joint/"))
    meas = ChannelMap(rec.channels, ("sensors/",))

    assert rec.channels["physics/body/body_frd"] is state["body_frd"]
    assert rec.channels["physics/joint/rotor_1_joint"] is state["rotor_1_joint"]
    assert set(state) == {"body_frd", "rotor_1_joint"}  # spans both prefixes; sensors excluded
    assert set(meas) == {"imu"}
    with pytest.raises(KeyError):
        state["imu"]  # not in body/ or joint/
