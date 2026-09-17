"""The Recorder's Rerun adapter: the teardown dump of every channel ring into the recording.

Real end-to-end in file mode, serve=False: the dump writes an inspectable ``.rrd``. Duck-typed over
the channel surface, ``fields`` / ``source`` / ``history_arrays``, so it needs no device ring. What
the dump *wrote* is read back through a stand-in sink, since rerun 0.34 dropped the local dataframe
API and the bundled command-line tool prints chunk metadata, not rows. Skipped if rerun is missing.
"""

import pytest

pytest.importorskip("rerun")


class _Counter:
    def numpy(self):
        import numpy as np

        return np.array([3])


class _FakeChannel:
    """Three recorded rows of a body channel: a three-column quantity and a scalar one."""

    fields = (("position", 3), ("q", 1))
    source = "NewtonPhysics"
    maxlen = 8
    dt = 0.004
    counter = _Counter()

    def history_arrays(self):
        import numpy as np

        t = np.arange(3) * 0.004
        return {"t": t, "position": np.ones((3, 3), dtype=np.float32), "q": np.zeros(3, dtype=np.float32)}


def test_dump_writes_one_series_entity_per_declared_quantity(tmp_path, rrd_entities):
    """Every declared field of every channel lands at ``recording/<channel key>/<field>``."""
    from nexus._src.logging import Logger
    from nexus._src.recording.rerun_adapter import dump

    rrd = str(tmp_path / "dump.rrd")
    rl = Logger(model=None, serve=False, record_to_rrd=rrd)
    dump({"physics/body/body_frd": _FakeChannel()}, rl)
    rl.close()

    paths = rrd_entities(rrd)
    assert {"/recording/physics/body/body_frd/position", "/recording/physics/body/body_frd/q"} <= set(paths), paths


def test_dump_draws_the_flown_path_from_the_airframe_channel(tmp_path, rrd_entities):
    """The trail is the base_body channel's recorded positions, not a producer's own accumulator."""
    from nexus._src.logging import Logger
    from nexus._src.recording.rerun_adapter import TRAJECTORY_ENTITY, dump

    rrd = str(tmp_path / "trail.rrd")
    rl = Logger(model=None, serve=False, record_to_rrd=rrd)
    dump({"physics/body/body_frd": _FakeChannel()}, rl, base_body=_FakeChannel())
    rl.close()

    assert f"/{TRAJECTORY_ENTITY}" in rrd_entities(rrd)


class _Sink:
    """A Logger stand-in that keeps what the dump handed it, the adapter's own output seam."""

    def __init__(self):
        self.trail = None
        self.tree = None

    def log_trail(self, entity, positions, times, *, color, stride=4):
        self.trail = (entity, positions, times)

    def show_recording(self, tree):
        self.tree = tree


def test_the_flown_path_carries_the_airframe_channel_positions():
    """Every trail point is the base_body channel's own recorded position, in its recorded order."""
    import numpy as np

    from nexus._src.recording.rerun_adapter import dump

    base_body = _FakeChannel()
    sink = _Sink()
    dump({"physics/body/body_frd": base_body}, sink, base_body=base_body)

    assert np.array_equal(sink.trail[1], base_body.history_arrays()["position"])


def test_the_flown_path_carries_the_recorded_times():
    """The trail rides the channel's own time column, so a point sits where the run flew it."""
    import numpy as np

    from nexus._src.recording.rerun_adapter import dump

    base_body = _FakeChannel()
    sink = _Sink()
    dump({"physics/body/body_frd": base_body}, sink, base_body=base_body)

    assert np.array_equal(sink.trail[2], base_body.history_arrays()["t"])
