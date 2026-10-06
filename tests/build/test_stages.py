"""The Proportional Integral Derivative (PID) example on the stage loop: the trajectory main recorded
and the seed row ``Sim.start()`` returns with. Auto-skips without a CUDA device; each test scopes the
device it uses, so the default device is the same after it.
"""

import hashlib

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("warp")

import warp as wp

from nexus_sim._src.api.sim import Sim
from nexus_sim._src.build.assembly import build_scenario
from nexus_sim._src.build.launch import resolve_to_vehicle_builder
from nexus_sim._src.config import LaunchConfig
from nexus_sim.examples.controllers.pid.assembly import build_pid_orchestrator

# The Secure Hash Algorithm (SHA) 256 digest of main's 200-tick PID body poses, float32 (200, 5, 7),
# per GPU model, flown through main's captured loop before the change: a CUDA trajectory repeats to the
# byte on one model and differs across models. A local run recorded the RTX 5080's, and a gpu-pytest
# run the A10G's. The same two runs recorded both again when the rotor speed at full command became
# the motor's no-load speed, a 32-bit float that reads 3e-8 below 3800 rpm.
_MAIN_TRAJECTORY = {
    "NVIDIA GeForce RTX 5080": "34dbc4aff792652d26d13262f94a88e56635906bedfd7030978037d970bbc186",
    "NVIDIA A10G": "c5ac89a9fdfe35221173d401a9c735969b8b97b6864ca97b24245e4c3723d603",
}


def _pid(*, cpu: bool, max_steps: int):
    cfg = build_scenario()
    cfg["physics"]["force_cpu"] = cpu
    vb, _ = resolve_to_vehicle_builder(LaunchConfig().set_vehicle("astro_max_base").set_scene("empty"))
    return build_pid_orchestrator(cfg, goal_w=(0.0, 0.0, 1.5), max_steps=max_steps, vehicle_builder=vb)


def test_pid_on_cuda_flies_the_trajectory_main_recorded():
    """The PID example on CUDA flies its whole tick as one graph, and its trajectory matches today's
    captured path: 200 ticks of body poses equal, to the byte, the ones main recorded through ``run()``
    before the change, on each GPU model main flew it on.
    """
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    gpu = wp.get_device("cuda:0").name
    if gpu not in _MAIN_TRAJECTORY:
        pytest.skip(f"main's trajectory was never recorded on {gpu!r}")
    with wp.ScopedDevice("cuda:0"):
        orch = _pid(cpu=False, max_steps=200)
        q = []
        orch.on_tick = lambda view, t, n: q.append(orch.physics.state0.body_q.numpy().copy())
        orch.run()
    body_q = np.ascontiguousarray(np.array(q, dtype=np.float32))
    assert hashlib.sha256(body_q.tobytes()).hexdigest() == _MAIN_TRAJECTORY[gpu]


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
