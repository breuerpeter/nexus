"""The recording holds every row, written from the Recorder's histories, #111.

Real end-to-end on the Warp CPU backend: a short flight of ``tests/recording/_flight.py``, with a
Recorder whose staging buffer holds 8 rows, so 50 ticks cross six drains, then the ``.rrd`` it wrote.
Skipped if rerun, newton or pxr are missing.
"""

import signal

import pytest

pytest.importorskip("rerun")
pytest.importorskip("newton")
pytest.importorskip("pxr")

import nexus_sim as nx
from tests.recording import _flight
from tests.recording._flight import DT, POSE, POSITION, rows, times
from tests.usd import sensor_vehicle as sv

pytestmark = pytest.mark.usefixtures("warp_cpu")

TICKS = 50


def _fly(orch, ticks: int) -> None:
    """Step the run ``ticks`` ticks through ``Sim``, then close it."""
    with nx.Sim.from_orchestrator(orch) as sim:
        for _ in range(ticks):
            assert sim.step()


def test_a_history_holds_every_row_of_a_run_with_no_step_cap(tmp_path):
    """A history holds every row of a run with no step cap, past the staging buffer and past today's
    30,000-row ring.

    Given a run of the fixture vehicle with no `max_steps` and a staging buffer of 8 rows, when it steps
    50 ticks, then the base body's history holds the start row and the 50 ticks' rows, oldest first, with
    `t` equal to `k * dt` for `k` in 0..50, and the newest row is the last.
    """
    orch, _ = _flight.build(tmp_path, staging=8)
    with nx.Sim.from_orchestrator(orch) as sim:
        for _ in range(TICKS):
            assert sim.step()
        history = sim.physics[sim.base_body].history()
        latest = sim.physics[sim.base_body].latest()

    assert ([s.t for s in history], latest.t) == (
        pytest.approx([k * DT for k in range(TICKS + 1)]),
        pytest.approx(TICKS * DT),
    )


def test_the_recording_holds_every_row_of_every_history_at_the_tick_rate(tmp_path):
    """The recording holds every row of every history, at the tick rate, not a decimated or capped subset.

    Given a recorded run of a fixture with one joint, a staging buffer of 8 rows, when it steps 50 ticks
    and ends, then each body and joint series in the `.rrd` holds the start row and the 50 ticks' rows,
    with `t` from `0` to `50 * dt`.
    """
    orch, rrd = _flight.build(tmp_path, staging=8, usd=sv.vehicle(tmp_path))
    _fly(orch, TICKS)

    expected = pytest.approx([k * DT for k in range(TICKS + 1)])
    assert (times(rrd, POSITION), times(rrd, "/sim/vehicle/joints/rotor_1_joint/q")) == (expected, expected)


def test_a_hard_kill_loses_at_most_one_block(tmp_path):
    """The recording gains each block as the staging buffer drains, so a `SIGKILL` loses at most one block.

    Given a recorded run in a subprocess with a staging buffer of 8 rows, when `SIGKILL` lands after 50
    ticks, then the `.rrd` opens and each history's series holds at least 40 rows.
    """
    flight = _flight.run_until_ready(tmp_path, signal.SIGKILL, mode="staged")

    assert rows(flight.rrd, POSITION) >= 40


def test_each_mesh_is_logged_once_and_each_bodys_pose_rides_the_histories_as_columns(tmp_path, rrd_rows):
    """The Logger logs each mesh once, and each body's pose rides the histories as columns.

    Given a recorded run of a fixture with a box shape, which NVIDIA Newton's viewer logs as a mesh,
    when it steps 50 ticks and ends, then the `.rrd` holds one static mesh row per shape, a transform
    row per history row at the base body's entity, the start and the 50 ticks, and no entity holds 51
    mesh rows.
    """
    orch, rrd = _flight.build(tmp_path, staging=8)
    _fly(orch, TICKS)

    meshes: dict[str, int] = {}
    for entity, _, columns in rrd_rows(rrd):
        if "Mesh3D:vertex_positions" in columns:
            meshes[entity] = meshes.get(entity, 0) + len(columns["Mesh3D:vertex_positions"])
    poses = sum(
        len(columns["Transform3D:translation"])
        for entity, _, columns in rrd_rows(rrd)
        if entity == POSE and "Transform3D:translation" in columns
    )

    assert (sorted(set(meshes.values())), poses) == ([1], TICKS + 1)
