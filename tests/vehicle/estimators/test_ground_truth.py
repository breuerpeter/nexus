"""The passthrough estimator: each reader of its estimate gets the base body's true pose and twist of the same
tick, from the pass before the first tick on. It flies the fixture quad of `tests/vehicle/quad.py` in free fall,
on real physics, so every tick's state differs from the last. Each test scopes the device it uses, so the default
device is the same after it.
"""

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import warp as wp

from nexus_sim._src.api.sim import Sim
from nexus_sim._src.build.assembly import build_scenario
from nexus_sim._src.core import Clock, Orchestrator
from nexus_sim._src.core.interfaces import Stage
from nexus_sim._src.core.schema import PoseTwist, PositionGoal
from nexus_sim._src.core.signals import Signal
from nexus_sim._src.guidance import MissionGuidance
from nexus_sim._src.physics import NewtonPhysics
from nexus_sim._src.physics.builders.usd import USDBuilder
from nexus_sim._src.vehicle.estimators import GroundTruthEstimator
from tests.vehicle import quad

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
TICKS = 5


class _KeepingGuidance(MissionGuidance):
    """A guidance that keeps each position it reads. Its one goal sits 50 m up, so its mission never ends."""

    def __init__(self):
        super().__init__()
        self.kept = []
        self.set_mission([(0.0, 0.0, 50.0)])

    def _tick(self, pos, ts):
        self.kept.append(np.array(pos))
        super()._tick(pos, ts)


class _KeepingController:
    """A controller whose host stage keeps each estimate it reads. It reads the guidance's setpoint too."""

    def __init__(self):
        self.estimate = Signal("estimate", PoseTwist, shape=(1, 13))
        self.setpoint = Signal("setpoint", PositionGoal, shape=(1,))
        self.kept = []

    def connect(self):
        pass

    def close(self):
        pass

    def stages(self):
        def keep(tick):
            self.kept.append(self.estimate.read()[0])
            return True

        return [Stage("keep", "host", keep, reads=(self.estimate, self.setpoint))]


def _bits(values) -> np.ndarray:
    """The values as 32-bit floats, viewed as their bits, so two arrays compare bit for bit."""
    return np.ascontiguousarray(np.array(values, dtype=np.float32)).view(np.uint32)


@pytest.mark.parametrize("device", DEVICES)
def test_each_reader_gets_the_base_bodys_true_pose_and_twist_with_no_noise_and_no_delay(device, tmp_path):
    """The passthrough estimator gives each reader the base body's true pose and twist in world axes, with no
    noise and no delay, from the pass before the first tick on.

    Given the fixture vehicle falling from 2 m on the passthrough estimator, a guidance that keeps each position
    it reads and a stand-in controller whose host stage keeps each estimate it reads, when the run starts and
    steps 5 ticks, eagerly on the CPU device and captured on a CUDA device, then each value they read equals, to
    the bit, the base body's row the Recorder holds for that tick, and the seed row in the pass before the first
    tick.
    """
    cfg = build_scenario()
    cfg["physics"]["force_cpu"] = device == "cpu"
    cfg["physics"]["spawn"] = {"pos": (0.0, 0.0, 2.0)}
    guidance = _KeepingGuidance()
    with wp.ScopedDevice(device):
        controller = _KeepingController()
        orch = Orchestrator(
            clock=Clock(cfg["physics"]["dt"]),
            physics=NewtonPhysics(
                vehicle_builder=USDBuilder({"usd_path": str(quad.author(tmp_path / "quad.usda"))}, None), cfg=cfg
            ),
            sensors=[],
            estimator=GroundTruthEstimator(),
            controller=controller,
            max_steps=TICKS,
        )
        with Sim.from_orchestrator(orch, guidance=guidance) as sim:
            sim.run()
            rows = sim.physics[sim.base_body].history()
    truth = [[*r.position, *r.quat_xyzw, *r.velocity, *r.angular_velocity] for r in rows]

    assert (
        np.array_equal(_bits(guidance.kept), _bits(truth)[:, :3]),
        np.array_equal(_bits(controller.kept), _bits(truth)),
    ) == (True, True)
