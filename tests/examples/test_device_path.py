"""The device-native control path, the capture and autodiff prerequisite: with the Warp signal ``observation``
the Proportional Integral Derivative (PID) loop has no per-tick host hop. WarpObservationSensor
-> PidController.exchange, the law + moment mixer, -> Rotors.forces_wp all stay on-device, so
the whole tick is one graph. A NumPy observation still takes the host path.

The single-body motor model + the moment mixer need rotor geometry for the allocation, so the fixture
builds a real rotored vehicle, the local fixture vehicle of ``tests/usd/sensor_vehicle.py``.
"""

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("warp")

import newton
import warp as wp

from nexus_sim._src.build.launch import resolve_vehicle_usd
from nexus_sim._src.config import LaunchConfig
from nexus_sim._src.core.schema import SimTime
from nexus_sim._src.core.signals import wire
from nexus_sim._src.core.stages import Bound
from nexus_sim.examples._lib import Rotors, build_rotor_mixer_from_model
from nexus_sim.examples._lib.observation import WarpObservationSensor
from nexus_sim.examples.controllers.pid import PidController
from tests.usd import sensor_vehicle as sv

pytestmark = pytest.mark.usefixtures("warp_cpu")

_ACT_CFG = {"ct": 0.000003463, "cd": 0.05, "rpm_max": 3800.0}  # the thrust map the fixture's rotors declare


def _is_warp_array(x) -> bool:
    return not isinstance(x, np.ndarray) and hasattr(x, "numpy")


def _rotored_model(tmp_path):
    """A real rotored vehicle, the fixture, + its settled rest pose: the geometry the allocation needs."""
    b = newton.ModelBuilder()
    vehicle_usd, _ = resolve_vehicle_usd(LaunchConfig().set_vehicle(sv.vehicle(tmp_path)).set_scene(sv.SCENE))
    vehicle_usd.build(b)
    model = b.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)  # populate body_q, the rotor offsets
    mass = float(np.sum(model.body_mass.numpy()))
    return model, vehicle_usd.rotor_joints(), state, mass


def test_a_warp_observation_keeps_the_pid_path_on_the_device(tmp_path):
    model, joints, state, mass = _rotored_model(tmp_path)
    mixer = build_rotor_mixer_from_model(model, joints, _ACT_CFG, state.body_q.numpy())

    # 1. obs sensor writes the signal observation, a Warp array that PID reads, not numpy
    sensor = WarpObservationSensor(goal_w=(0.0, 0.0, 1.5))
    pid = PidController(goal_w=(0.0, 0.0, 1.5), thrust_to_weight=1.9, mixer=mixer, weight=mass * 9.81)
    pairs = ((sensor, "sensor"), (pid, "controller"))
    wire([Bound(stage, component, role) for component, role in pairs for stage in component.stages()])
    sensor.sample_wp(state, sensor.out.buffer)
    assert _is_warp_array(pid.observation.buffer)
    assert pid.observation.read().shape == (12,)

    # 2. exchange runs the law + moment mixer on-device -> Warp Controls = nr per-rotor commands, no host hop
    controls = pid.exchange(SimTime(0.0, 0))
    assert _is_warp_array(controls.command)
    assert controls.command.numpy().reshape(-1).shape == (mixer.nr,)

    # 3. the actuator's device stage applies the Warp command buffer directly, no H2D, and writes a real wrench
    act = Rotors(mixer=mixer, dt=0.004, thrust_sign=-1.0, motor_tau=0.033)
    state.clear_forces()
    act.forces_wp(controls.command, state)
    wp.synchronize()
    bf = state.body_f.numpy()[act.base]
    assert np.isfinite(bf).all() and np.any(bf != 0.0)


def test_numpy_observation_takes_host_path():
    pid = PidController(
        goal_w=(0.0, 0.0, 1.5), thrust_to_weight=1.9
    )  # law-only, no mixer: the host path returns the raw moments
    # The provider's state builds a NumPy observation: level, at rest, at the goal's height.
    pid.bind_state_provider(lambda: ((0.0, 0.0, 1.5), (1.0, 0.0, 0.0, 0.0), (0, 0, 0), (0, 0, 0)))
    controls = pid.exchange(SimTime(0.0, 0))
    assert isinstance(controls.command, np.ndarray)  # host path preserved: numpy per-rotor commands
    assert controls.command.shape == (4,)
