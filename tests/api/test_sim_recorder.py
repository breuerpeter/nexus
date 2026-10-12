"""`Sim` takes no `observe`, and the Recorder is on for every run, #111.

Real on the Warp CPU backend: a short flight of ``tests/recording/_flight.py`` with no Logger, read
through ``Sim``. Skipped if newton or pxr are missing.
"""

import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import nexus_sim as nx
from tests.recording import _flight

pytestmark = pytest.mark.usefixtures("warp_cpu")


def _unrecorded(tmp_path):
    """A run with no Logger, as `Sim(log=False)` builds one."""
    orch, _ = _flight.build(tmp_path, staging=None)
    orch.logger = None
    return orch


def test_sim_takes_no_observe_argument():
    """`Sim` takes no `observe`.

    Given `Sim(..., observe=False)`, when a script builds it, then it raises `TypeError`; and so does
    `Sim._from_loop(..., observe=False)`.
    """
    with pytest.raises(TypeError, match="observe"):
        nx.Sim("astro_max_base", scene="empty", device="cpu", observe=False)
    with pytest.raises(TypeError, match="observe"):
        nx.Sim._from_loop(object(), observe=False)


def test_a_run_with_no_recording_still_reads_its_physics(tmp_path):
    """The Recorder is on for every run, `log=False` included.

    Given a `Sim` with `log=False`, when it steps once, then the base body's newest row
    is that tick's row: the body, spawned in free fall, has fallen.
    """
    with nx.Sim._from_loop(_unrecorded(tmp_path)) as sim:
        sim.step()
        latest = sim[f"vehicle/body/{sim.base_body}"].history()[-1]

    assert latest["linear_velocity"][2] < 0


def test_a_run_with_no_recording_still_waits_on_its_physics(tmp_path):
    """The Recorder is on for every run, `log=False` included.

    Given a `Sim` with `log=False`, when a script waits until the body has fallen 0.1 m, then
    `wait_until` returns.
    """
    with nx.Sim._from_loop(_unrecorded(tmp_path)) as sim:
        sim.wait_until(lambda: sim.read(f"vehicle/body/{sim.base_body}").position[2] < 4.9, sim_timeout=2.0)
        fell_to = sim[f"vehicle/body/{sim.base_body}"].history()[-1]["position"][2]

    assert fell_to < 4.9
