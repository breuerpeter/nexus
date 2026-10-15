"""Every example flies the articulated plant on Newton's motors: its vehicle keeps its rotor joints, and each
turns under its Newton motor. The runs of the Proportional Integral Derivative (PID), policy and sampling Model
Predictive Control (MPC) examples, and the rollouts of the two gradient examples, on the Newton CPU backend;
the build's force_cpu sets the device, and the scope puts it back.
"""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("newton")

import torch

from nexus_uav._src.api.sim import Sim
from nexus_uav._src.build.launch import resolve_scenario, resolve_vehicle_usd
from nexus_uav._src.config import LaunchConfig
from nexus_uav._src.guidance import MissionGuidance
from nexus_uav.examples.controllers.pid.assembly import build_pid_loop
from nexus_uav.examples.controllers.policy.assembly import build_policy_loop
from nexus_uav.examples.controllers.policy.goto.geofence import GeofenceGuidance
from nexus_uav.examples.controllers.sampling_mpc.assembly import build_sampling_mpc_loop
from nexus_uav.examples.design_opt import gain_tuning, mass_recovery
from tests.usd import sensor_vehicle as sv

TICKS = 50  # 0.2 s at the 4 ms tick, long past a motor's spin-up of about 8 ms
# A rotor that turns under its motor: hovering takes about 267 rad/s on `astro_max_base`, and a free rotor
# joint turns at none.
TURNING = 100.0  # rad/s


def _cpu() -> dict:
    cfg = sv.scenario()
    cfg["physics"]["force_cpu"] = True
    return cfg


def _flown(loop, guidance, mission) -> list[float]:
    """Fly `loop` toward `mission` for its ticks, and return each rotor joint's speed at the last tick, in rad/s."""
    with Sim._from_loop(loop, guidance=guidance) as sim:
        guidance.set_mission(mission)
        sim.run()
    rotors = [path for path in sim if path.startswith("vehicle/joints/") and "rotor" in path]
    return [abs(float(sim[path].history()[-1]["qd"][0])) for path in rotors]


def _pid(tmp_path) -> list[float]:
    """The PID example's run toward 1.5 m up."""
    vb, _ = resolve_vehicle_usd(LaunchConfig().set_vehicle("astro_max_base").set_scene(sv.SCENE))
    loop = build_pid_loop(_cpu(), goal_w=(0.0, 0.0, 1.5), max_steps=TICKS, vehicle_usd=vb)
    return _flown(loop, MissionGuidance(), [(0.0, 0.0, 1.5)])


class _Hold(torch.nn.Module):
    """A policy that holds the collective at its middle and asks for no body rate."""

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(obs.shape[0], 4)


def _policy(tmp_path) -> list[float]:
    """The policy example's run, flying a policy that holds the collective at its middle."""
    path = tmp_path / "hold.pt"
    torch.jit.script(_Hold()).save(str(path))
    vb, _ = resolve_vehicle_usd(LaunchConfig().set_vehicle("astro_max_base").set_scene("empty"))
    loop = build_policy_loop(_cpu(), policy_path=str(path), vehicle_usd=vb, max_steps=TICKS)
    return _flown(loop, GeofenceGuidance(bounds=((-3.0, -3.0, -1.0), (3.0, 3.0, 4.0))), [(0.0, 0.0, 1.0)])


def _sampling_mpc(tmp_path) -> list[float]:
    """The sampling MPC example's run in its slalom, toward 1.5 m up."""
    launch = LaunchConfig().set_vehicle("astro_max_base").set_scene("slalom")
    launch.runtime.device = "cpu"
    vb, _, cfg = resolve_scenario(launch)
    cfg["physics"]["force_cpu"] = True
    loop = build_sampling_mpc_loop(cfg, vehicle_usd=vb, num_rollouts=2, max_steps=TICKS)
    return _flown(loop, MissionGuidance(), [(0.0, 0.0, 1.5)])


def _gain_tuning(tmp_path) -> list[float]:
    """The gain tuning example's rollout, flying its stable reference gains."""
    rollout = gain_tuning.AstroMaxWaypointRollout(n_steps=TICKS)
    rollout.trajectory(np.array([0.3, 1.0, 0.6, 1.2, 8.0, 3.0, 1.0], dtype=np.float32))
    model = rollout.model
    starts = model.joint_qd_start.numpy()
    dofs = [int(starts[j]) for j, label in enumerate(model.joint_label) if "rotor" in label]
    qd = rollout.states[-1].joint_qd.numpy()
    return [abs(float(qd[d])) for d in dofs]


def _mass_recovery(tmp_path) -> list[float]:
    """The mass recovery example's rollout, over one step of its descent."""
    recovered = mass_recovery.recover_mass(n_steps=TICKS, iters=1)
    return [abs(float(w)) for w in recovered.get("rotor_speeds", [])]


@pytest.mark.usefixtures("warp_cpu")
@pytest.mark.parametrize(
    "run",
    [_pid, _policy, _sampling_mpc, _gain_tuning, _mass_recovery],
    ids=["pid", "goto_policy", "sampling_mpc", "gain_tuning", "mass_recovery"],
)
def test_every_example_flies_the_articulated_plant_on_newtons_motors(run, tmp_path):
    """Every example flies the articulated plant on Newton's motors.

    Given each of `pid`, `goto_policy`, `sampling_mpc`, `gain_tuning` and `mass_recovery`, when it runs, then
    its vehicle keeps its rotor joints and each turns under its Newton motor.
    """
    speeds = run(tmp_path)

    assert (len(speeds), min(speeds, default=0.0) > TURNING) == (4, True)
