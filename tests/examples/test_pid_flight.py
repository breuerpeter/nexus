"""The Proportional Integral Derivative (PID) example flies the shipped vehicle: a new vehicle Universal Scene
Description (USD) file changes nothing in its flight.

Heavy, since it builds and flies a Newton model on CPU; skipped if ``newton`` isn't importable.
"""

import pytest

pytest.importorskip("newton")
pytest.importorskip("warp")

import nexus as na
from nexus._src.build.launch import resolve_scenario
from nexus._src.config import LaunchConfig
from nexus.examples.controllers.pid import flight
from nexus.examples.controllers.pid.assembly import build_pid_orchestrator

pytestmark = pytest.mark.usefixtures("warp_cpu")  # the build's force_cpu sets the device; the scope puts it back


def test_the_pid_example_flies_the_shipped_vehicle_to_its_first_waypoint():
    """The example controllers fly the re-authored vehicles as before.

    Given the shipped `astro_max_base`, when the example flies its tuned gains on CPU toward the first
    waypoint of its tour, then the operator reports that waypoint reached.
    """
    launch = LaunchConfig().set_vehicle(flight.VEHICLE).set_scene(flight.SCENE)
    launch.runtime.device = "cpu"
    launch.runtime.solver = "semi_implicit"
    builder, _, cfg = resolve_scenario(launch)
    orch = build_pid_orchestrator(
        cfg,
        vehicle_builder=builder,
        goal_w=flight.WAYPOINTS[0],
        gains=flight.GAINS,
        moment_scale=flight.MOMENT_SCALE,
        max_steps=3000,
    )
    with na.Sim.from_orchestrator(orch, reached_m=0.3, final_hold_s=0.0) as sim:
        sim.operator.set_mission(flight.WAYPOINTS[:1])
        sim.run()

    assert sim.operator.reached == 1
