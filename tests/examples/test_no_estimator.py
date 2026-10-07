"""A guidance or a controller that reads the estimate needs an estimator: a run with none fails before any
stage runs, and the error names the reader and the missing estimator. The runs of the policy, the sampling
Model Predictive Control (MPC) and the Proportional Integral Derivative (PID) examples, on the Newton CPU
backend; the build's force_cpu sets the device, and the scope puts it back.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("newton")

from nexus_sim._src.api.sim import Sim
from nexus_sim._src.build.assembly import build_scenario
from nexus_sim._src.build.launch import resolve_scenario, resolve_to_vehicle_builder
from nexus_sim._src.config import LaunchConfig
from nexus_sim._src.guidance import MissionGuidance
from nexus_sim.examples.controllers.pid.assembly import build_pid_orchestrator
from nexus_sim.examples.controllers.policy.assembly import build_policy_orchestrator
from nexus_sim.examples.controllers.sampling_mpc.assembly import build_sampling_mpc_orchestrator
from tests.usd import sensor_vehicle as sv


def _cpu() -> dict:
    cfg = build_scenario()
    cfg["physics"]["force_cpu"] = True
    return cfg


def _policy(torchscript_policy):
    """The policy example's run, with no guidance."""
    vb, _ = resolve_to_vehicle_builder(LaunchConfig().set_vehicle("astro_max_base").set_scene("empty"))
    return build_policy_orchestrator(_cpu(), policy_path=torchscript_policy(), vehicle_builder=vb, max_steps=1), None


def _sampling_mpc(torchscript_policy):
    """The sampling MPC example's run in its slalom, with no guidance."""
    launch = LaunchConfig().set_vehicle("astro_max_base").set_scene("slalom")
    launch.runtime.device = "cpu"
    vb, _, cfg = resolve_scenario(launch)
    cfg["physics"]["force_cpu"] = True
    return build_sampling_mpc_orchestrator(cfg, vehicle_builder=vb, num_rollouts=2, max_steps=1), None


def _pid_with_guidance(torchscript_policy):
    """The PID example's run, with a guidance toward 1.5 m up."""
    vb, _ = resolve_to_vehicle_builder(LaunchConfig().set_vehicle("astro_max_base").set_scene(sv.SCENE))
    guidance = MissionGuidance()
    guidance.set_mission([(0.0, 0.0, 1.5)])
    return build_pid_orchestrator(_cpu(), goal_w=(0.0, 0.0, 1.5), max_steps=1, vehicle_builder=vb), guidance


@pytest.mark.usefixtures("warp_cpu")
@pytest.mark.parametrize(
    ("build", "reader"),
    [
        (_policy, "TrainedPolicyController"),
        (_sampling_mpc, "SamplingMPCController"),
        (_pid_with_guidance, "MissionGuidance"),
    ],
    ids=["policy", "sampling_mpc", "pid_with_guidance"],
)
def test_a_reader_of_the_estimate_on_a_run_with_no_estimator_fails_before_any_stage_runs(
    build, reader, torchscript_policy
):
    """A run whose guidance or controller reads the estimate fails before any stage runs when no estimator
    writes it, and the error says so.

    Given the policy example's run and the sampling MPC example's run, each with no estimator and no guidance,
    when the run starts, then it fails before any stage runs, and the error names its controller,
    `TrainedPolicyController` or `SamplingMPCController`, and says no estimator writes the estimate; once more
    with the PID example's run, a `MissionGuidance` and no estimator, where the error names `MissionGuidance`.
    No stage ran, so the recording holds no row.
    """
    orch, guidance = build(torchscript_policy)
    orch.estimator = None
    with Sim.from_orchestrator(orch, guidance=guidance) as sim:
        with pytest.raises(ValueError) as e:
            sim.run()
        rows = len(sim.physics[sim.base_body].history())
    assert (reader in str(e.value), "estimator" in str(e.value), rows) == (True, True, 0)
