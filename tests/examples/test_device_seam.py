"""The device-native control seam, the capture/autodiff prerequisite: with a Warp ``observation`` slot
the in-process Proportional Integral Derivative (PID) loop has no per-tick host hop. WarpObservationSensor
-> PidController.exchange, the law + moment mixer, -> RigidBodyRotors.forces all stay on-device, and the
actuator is ``capturable``. A NumPy observation still takes the host path.

The single-body motor model + the moment mixer need rotor geometry for the allocation, so the seam fixture
builds a real rotored vehicle, the astro-max Universal Scene Description (USD).
"""

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("warp")

import newton
import warp as wp

from nexus._src.config import LaunchConfig
from nexus._src.core.schema import Measurement, SimTime
from nexus._src.runtimes.launch import resolve_to_vehicle_builder
from nexus.examples._lib import RigidBodyRotors, build_rotor_mixer_from_model
from nexus.examples._lib.observation import WarpObservationSensor
from nexus.examples.controllers.pid import PidController

pytestmark = pytest.mark.usefixtures("warp_cpu")

_ACT_CFG = {"ct": 0.000003463, "cd": 0.05, "rpm_max": 3800.0}  # astro-max thrust map, from freefly:actuator:*


def _is_warp_array(x) -> bool:
    return not isinstance(x, np.ndarray) and hasattr(x, "numpy")


def _rotored_model():
    """A real rotored vehicle, astro-max, + its settled rest pose: the geometry the allocation needs."""
    b = newton.ModelBuilder()
    vb, _ = resolve_to_vehicle_builder(LaunchConfig().set_vehicle("astro_max_base"))
    vb.build(b)
    model = b.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)  # populate body_q, the rotor offsets
    mass = float(np.sum(model.body_mass.numpy()))
    return model, state, mass


def test_inprocess_seam_is_device_native():
    model, state, mass = _rotored_model()
    mixer = build_rotor_mixer_from_model(model, _ACT_CFG, state.body_q.numpy())

    # 1. obs sensor writes a Warp array into meas.observation, not numpy
    meas = Measurement()
    WarpObservationSensor(goal_w=(0.0, 0.0, 1.5)).sample(state, None, SimTime(0.0, 0), meas)
    assert _is_warp_array(meas.observation)
    assert meas.observation.numpy().shape == (12,)

    # 2. exchange runs the law + moment mixer on-device -> Warp Controls = nr per-rotor commands, no host hop
    pid = PidController(goal_w=(0.0, 0.0, 1.5), thrust_to_weight=1.9, mixer=mixer, weight=mass * 9.81)
    controls = pid.exchange(meas, SimTime(0.0, 0))
    assert _is_warp_array(controls.command)
    assert controls.command.numpy().reshape(-1).shape == (mixer.nr,)

    # 3. the actuator applies the Warp Controls directly, no H2D, and writes a real wrench; it's capturable
    act = RigidBodyRotors(mixer=mixer, dt=0.004, thrust_sign=-1.0, motor_tau=0.033)
    assert act.capturable  # no per-tick host op, the determinism path -> the in-process loop joins a graph
    state.clear_forces()
    act.forces(controls, state, None)
    wp.synchronize()
    bf = state.body_f.numpy()[act.base]
    assert np.isfinite(bf).all() and np.any(bf != 0.0)


def test_numpy_observation_takes_host_path():
    pid = PidController(
        goal_w=(0.0, 0.0, 1.5), thrust_to_weight=1.9
    )  # law-only, no mixer: the host path returns the raw moments
    meas = Measurement()
    meas.observation = np.zeros(12, np.float32)
    meas.observation[6:9] = [0.0, 0.0, -1.0]  # level gravity
    controls = pid.exchange(meas, SimTime(0.0, 0))
    assert isinstance(controls.command, np.ndarray)  # host path preserved: numpy per-rotor commands
    assert controls.command.shape == (4,)
