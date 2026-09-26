"""The Proportional Integral Derivative (PID) example on the stage loop: the trajectory main recorded
and the seed row ``Sim.start()`` returns with. Auto-skips without a CUDA device; each test scopes the
device it uses, so the default device is the same after it.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("warp")

import warp as wp

from nexus._src.api.sim import Sim
from nexus._src.build.assembly import build_scenario
from nexus._src.build.launch import resolve_to_vehicle_builder
from nexus._src.config import LaunchConfig
from nexus.examples.controllers.pid.assembly import build_pid_orchestrator

_REFERENCE = Path(__file__).parent / "data" / "pid_cuda_body_q_200.npz"


def _pid(*, cpu: bool, max_steps: int):
    cfg = build_scenario()
    cfg["physics"]["force_cpu"] = cpu
    vb, _ = resolve_to_vehicle_builder(LaunchConfig().set_vehicle("astro_max_base"))
    return build_pid_orchestrator(cfg, goal_w=(0.0, 0.0, 1.5), max_steps=max_steps, vehicle_builder=vb)


def test_pid_on_cuda_flies_the_trajectory_main_recorded():
    """The PID example on CUDA flies its whole tick as one graph, and its trajectory matches today's
    captured path: 200 ticks of body poses equal, to the byte, the ones main recorded through ``run()``
    before the change, on an RTX 5080.
    """
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    with wp.ScopedDevice("cuda:0"):
        orch = _pid(cpu=False, max_steps=200)
        q = []
        orch.on_tick = lambda view, t, n: q.append(orch.physics.state0.body_q.numpy().copy())
        orch.run()
    assert np.array_equal(np.array(q), np.load(_REFERENCE)["body_q"])


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_start_returns_with_the_seed_row_and_the_first_tick_recorded(device):
    """``Sim.start()`` returns with one observation row recorded and the capture done, for every
    controller kind: the pid example's base body holds two rows after ``start()``, the settled seed row
    and the first tick's.
    """
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    with wp.ScopedDevice(device), Sim.from_orchestrator(_pid(cpu=device == "cpu", max_steps=50)) as sim:
        sim.start()
        rows = len(sim.physics[sim.base_body].history())
    assert rows == 2
