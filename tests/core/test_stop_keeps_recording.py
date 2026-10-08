"""Every stop that reaches Python writes the recording, #111.

Real end-to-end on the Warp CPU backend: a short flight of ``tests/recording/_flight.py``, ended by a
component that raises, or run as a script and ended by a signal, then the ``.rrd`` it wrote. Skipped if
rerun, newton or pxr are missing.
"""

import signal

import pytest

pytest.importorskip("rerun")
pytest.importorskip("newton")
pytest.importorskip("pxr")

import nexus_sim as nx
from tests.recording import _flight
from tests.recording._flight import POSITION, rows

pytestmark = pytest.mark.usefixtures("warp_cpu")

TICKS = 50


class _Peer:
    """A peer the build started, whose stop fails as a container's does without a docker daemon."""

    def start(self):
        raise AssertionError("the loop starts no peer; the build does")

    def stop(self):
        raise RuntimeError("docker daemon went away")

    def alive(self):
        return True


def test_teardown_writes_the_last_block_and_closes_the_file_before_the_controller_closes_and_the_peers_stop(
    tmp_path,
):
    """Teardown writes the last block and closes the file before the controller closes and the peers stop.

    Given a recorded run whose controller's `close` raises and whose peer's `stop` raises, when it ends
    after 50 ticks, then the `.rrd` holds all 51 rows of the base body's history, the start and the 50
    ticks. The controller reads the `.rrd` as its `close` runs, so it reads the rows written before it.
    """
    seen: list[int] = []

    class _Controller(_flight.Controller):
        def close(self):
            seen.append(rows(rrd, POSITION))
            raise RuntimeError("docker daemon went away")

    orch, rrd = _flight.build(tmp_path, staging=8, max_steps=TICKS, controller=_Controller(), peers=[_Peer()])
    with pytest.raises(RuntimeError, match="docker daemon went away"):
        orch.run()

    assert seen == [TICKS + 1]


def test_a_python_exception_in_a_stage_still_writes_the_recording(tmp_path):
    """A Python exception in a stage still writes the recording.

    Given a recorded run whose controller's stage raises on tick 20, when the run ends, then the
    exception propagates and the `.rrd` holds the 20 rows before it, the start and 19 ticks.
    """

    def act(tick):
        if tick.t.step_index == 20:
            raise ValueError("the controller's stage failed")

    orch, rrd = _flight.build(tmp_path, staging=None, controller=_flight.Controller(act))
    with nx.Sim.from_orchestrator(orch) as sim, pytest.raises(ValueError, match="stage failed"):
        sim.run()

    assert rows(rrd, POSITION) == 20


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP], ids=["SIGTERM", "SIGHUP"])
def test_sigterm_and_sighup_stop_a_run_as_ctrl_c_does_and_the_recording_holds_every_row(tmp_path, sig):
    """SIGTERM and SIGHUP stop a run as Ctrl-C does, and the recording holds every row.

    Given a recorded run in a subprocess, when `SIGTERM`, and in a second run `SIGHUP`, lands after 50
    ticks, then the process exits 0 within 10 s and the `.rrd` holds every row of the base body's
    history. The script catches the `KeyboardInterrupt` a Ctrl-C raises and exits 0 on it, and prints
    the rows its history holds as it lands.
    """
    flight = _flight.run_until_ready(tmp_path, sig, timeout=10.0)

    assert (flight.returncode, rows(flight.rrd, POSITION)) == (0, flight.history)


def test_a_ctrl_c_during_the_write_waits_until_the_file_is_closed(tmp_path):
    """A Ctrl-C during the write waits until the file closes.

    Given a recorded run in a subprocess with a write slowed to 2 s, when `SIGINT` lands after 50 ticks
    and a second `SIGINT` lands during the write, then the process exits and the `.rrd` opens with every
    row of the base body's history.
    """
    flight = _flight.run_until_ready(tmp_path, signal.SIGINT, mode="slow-write", then=signal.SIGINT, timeout=20.0)

    assert (flight.returncode, rows(flight.rrd, POSITION)) == (0, flight.history)
