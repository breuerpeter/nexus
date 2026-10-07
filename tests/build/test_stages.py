"""The Proportional Integral Derivative (PID) example on the stage loop: the trajectory main recorded,
the seed row ``Sim.start()`` returns with, and the run a guidance of another setpoint type stops. The
CUDA tests are `gpu`; each test scopes the device it uses, so the default device is the same after it.
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
from nexus_sim._src.guidance import TrackingGuidance
from nexus_sim.examples.controllers.pid.assembly import build_pid_orchestrator
from tests.usd import sensor_vehicle as sv

# The Secure Hash Algorithm (SHA) 256 digest of main's 200-tick PID body poses, float32 (200, 5, 7),
# per GPU model, flown through main's captured loop before the change: a CUDA trajectory repeats to the
# byte on one model and differs across models. A local run recorded the RTX 5080's, and a gpu-pytest
# run the A10G's. The same two runs recorded both again when the rotor speed at full command became
# the motor's no-load speed, a 32-bit float that reads 3e-8 below 3800 rpm.
_MAIN_TRAJECTORY = {
    "NVIDIA GeForce RTX 5080": "34dbc4aff792652d26d13262f94a88e56635906bedfd7030978037d970bbc186",
    "NVIDIA A10G": "c5ac89a9fdfe35221173d401a9c735969b8b97b6864ca97b24245e4c3723d603",
}


def _pid(*, cpu: bool, max_steps: int, vehicle: str = "astro_max_base"):
    """The PID example's orchestrator flying `vehicle`, a catalog name or a path, toward 1.5 m up."""
    cfg = build_scenario()
    cfg["physics"]["force_cpu"] = cpu
    vb, _ = resolve_to_vehicle_builder(LaunchConfig().set_vehicle(vehicle).set_scene(sv.SCENE))
    return build_pid_orchestrator(cfg, goal_w=(0.0, 0.0, 1.5), max_steps=max_steps, vehicle_builder=vb)


@pytest.mark.gpu
def test_pid_on_cuda_flies_the_trajectory_main_recorded():
    """The PID example on CUDA flies its whole tick as one graph, and its trajectory matches today's
    captured path: 200 ticks of body poses equal, to the byte, the ones main recorded through ``run()``
    before the change, on each GPU model main flew it on.
    """
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


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def test_start_returns_with_the_seed_row_and_the_first_tick_recorded(device, tmp_path):
    """``Sim.start()`` returns with one observation row recorded and the capture done, for every
    controller kind: the pid example's base body holds two rows after ``start()``, the settled seed row
    and the first tick's. It flies the local fixture vehicle of ``tests/usd/sensor_vehicle.py``.
    """
    with (
        wp.ScopedDevice(device),
        Sim.from_orchestrator(_pid(cpu=device == "cpu", max_steps=50, vehicle=sv.vehicle(tmp_path))) as sim,
    ):
        sim.start()
        rows = len(sim.physics[sim.base_body].history())
    assert rows == 2


class _Reference:
    """A planned reference, the shape a tracking guidance writes."""

    duration = 4.0

    def set_start(self, p0):
        pass

    def reference_path(self):
        return []


def test_a_guidance_whose_setpoint_type_its_controller_does_not_read_fails_the_run_naming_both(tmp_path):
    """A guidance that writes a setpoint of another type than its controller reads fails the run before any
    stage runs, and the error names both components and both types.

    Given the PID example's run and a `TrackingGuidance`, which writes a `ReferenceTrajectory`, when the run
    starts, then it fails before any stage runs, and the error names `TrackingGuidance`, `PidController`,
    `ReferenceTrajectory` and `PositionGoal`. No stage ran, so the recording holds no row.
    """
    guidance = TrackingGuidance(planner=lambda waypoints: _Reference())
    guidance.set_mission([(1.0, 0.0, 1.5)])
    with (
        wp.ScopedDevice("cpu"),
        Sim.from_orchestrator(_pid(cpu=True, max_steps=50, vehicle=sv.vehicle(tmp_path)), guidance=guidance) as sim,
    ):
        with pytest.raises(ValueError) as e:
            sim.run()
        rows = len(sim.physics[sim.base_body].history())
    words = ("TrackingGuidance", "PidController", "ReferenceTrajectory", "PositionGoal")
    assert (all(word in str(e.value) for word in words), rows) == (True, 0)
