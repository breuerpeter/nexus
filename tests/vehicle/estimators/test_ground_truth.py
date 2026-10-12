"""The passthrough estimator: each reader of its estimate gets the base body's true pose and twist of the same
tick, from the pass before the first tick on. The run flies the fixture quad of `tests/vehicle/quad.py` in free
fall, on real physics, so every tick's state differs from the last. Each test scopes the device it uses, so the
default device is the same after it.
"""

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import warp as wp

from nexus_sim._src.api.sim import Sim
from nexus_sim._src.core import Clock, Loop
from nexus_sim._src.core.interfaces import Component, Stage, Tick
from nexus_sim._src.core.schema import PoseTwist, SimTime
from nexus_sim._src.core.signals import Signal, wire
from nexus_sim._src.core.stages import Bound
from nexus_sim._src.guidance import MissionGuidance
from nexus_sim._src.physics import NewtonPhysics
from nexus_sim._src.physics.vehicle import VehicleUsd
from nexus_sim._src.vehicle.estimators import GroundTruthEstimator
from tests.usd.sensor_vehicle import scenario
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


class _KeepingController(Component):
    """A controller whose host stage keeps each estimate it reads. It reads the guidance's setpoint too."""

    def __init__(self):
        self.estimate = Signal("estimate", PoseTwist, shape=(1,))
        self.setpoint = Signal("setpoint", wp.vec3, shape=(1,))
        self.kept = []

    def connect(self, links):
        pass

    def close(self):
        pass

    def stages(self):
        def keep(tick):
            self.kept.append(_row(self.estimate.read()[0]))
            return True

        return [Stage("keep", "host", keep, reads=(self.estimate, self.setpoint))]


def _row(estimate) -> np.ndarray:
    """An estimate as a body's row in the Recorder: position, quaternion, linear and angular velocity."""
    fields = ("position", "orientation", "linear_velocity", "angular_velocity")
    return np.concatenate([estimate[name] for name in fields])


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
    cfg = scenario()
    cfg["physics"]["force_cpu"] = device == "cpu"
    cfg["physics"]["spawn"] = {"pos": (0.0, 0.0, 2.0)}
    guidance = _KeepingGuidance()
    with wp.ScopedDevice(device):
        controller = _KeepingController()
        orch = Loop(
            clock=Clock(cfg["physics"]["dt"]),
            physics=NewtonPhysics(
                vehicle_usd=VehicleUsd({"usd_path": str(quad.author(tmp_path / "quad.usda"))}), cfg=cfg
            ),
            sensors=[],
            estimator=GroundTruthEstimator(),
            controller=controller,
            max_steps=TICKS,
        )
        with Sim._from_loop(orch, guidance=guidance) as sim:
            sim.run()
            rows = sim[f"vehicle/body/{sim.base_body}"].history()
    truth = [[*r["position"], *r["orientation"], *r["linear_velocity"], *r["angular_velocity"]] for r in rows]

    assert (
        np.array_equal(_bits(guidance.kept), _bits(truth)[:, :3]),
        np.array_equal(_bits(controller.kept), _bits(truth)),
    ) == (True, True)


class _TwoBodies(Component):
    """A stand-in plant whose `state` stage writes two bodies' poses, `body/a` and `body/b`, each its own."""

    def __init__(self):
        self.a = Signal("body/a", PoseTwist, shape=(1,))
        self.b = Signal("body/b", PoseTwist, shape=(1,))

    def stages(self):
        return [Stage("state", "device", lambda tick: None, writes=(self.a, self.b))]


@pytest.mark.usefixtures("warp_cpu")
def test_the_passthrough_writes_the_pose_its_input_holds_not_another_bodys():
    """The passthrough writes the pose and twist of the body a connection wires its `pose` input to, not another's.

    Given a stand-in plant that writes two bodies' poses and the passthrough estimator whose `pose` input a
    connection wires to the second body's output, when the plant writes and the passthrough's stage runs,
    then its estimate holds the second body's position, quaternion, linear velocity and angular velocity.
    """
    plant, estimator = _TwoBodies(), GroundTruthEstimator()
    stage = estimator.stages()[0]
    ring = [Bound(plant.stages()[0], plant, "physics"), Bound(stage, estimator, "estimator")]
    wire(ring, [(estimator, "pose", plant, "body/b")])
    plant.a.write([((9.0, 9.0, 9.0), (0.0, 0.0, 0.0, 1.0), (9.0, 9.0, 9.0), (9.0, 9.0, 9.0))])
    plant.b.write([((1.0, 2.0, 3.0), (0.5, -0.5, 0.5, 0.5), (4.0, 5.0, 6.0), (7.0, 8.0, 9.5))])
    stage.run(Tick(state=None, t=SimTime(), dt=0.004))

    row = _row(estimator.estimate.read()[0])

    assert row.tolist() == [1.0, 2.0, 3.0, 0.5, -0.5, 0.5, 0.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.5]
