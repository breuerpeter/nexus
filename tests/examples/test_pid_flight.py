"""The Proportional Integral Derivative (PID) example flies the shipped vehicle: a new vehicle Universal Scene
Description (USD) file changes nothing in its flight.

Heavy, since it builds and flies a Newton model on CPU; skipped if ``newton`` isn't importable.
"""

import pytest

pytest.importorskip("newton")
pytest.importorskip("warp")

import numpy as np

import nexus_uav as nx
from nexus_uav._src.build.launch import resolve_scenario
from nexus_uav._src.config import LaunchConfig
from nexus_uav._src.guidance import MissionGuidance
from nexus_uav.examples.controllers.pid import flight
from nexus_uav.examples.controllers.pid.assembly import build_pid_loop
from tests.vehicle import quad

pytestmark = pytest.mark.usefixtures("warp_cpu")  # the build's force_cpu sets the device; the scope puts it back


def test_the_pid_example_flies_the_shipped_vehicle_to_its_first_waypoint():
    """The example controllers fly the re-authored vehicles as before.

    Given the shipped `astro_max_base`, when the example flies its tuned gains on CPU toward the first
    waypoint of its tour, then the guidance reports that waypoint reached.
    """
    launch = LaunchConfig().set_vehicle(flight.VEHICLE).set_scene(flight.SCENE)
    launch.runtime.device = "cpu"
    vehicle_usd, _, cfg = resolve_scenario(launch)
    orch = build_pid_loop(
        cfg,
        vehicle_usd=vehicle_usd,
        goal_w=flight.WAYPOINTS[0],
        gains=flight.GAINS,
        moment_scale=flight.MOMENT_SCALE,
        max_steps=3000,
    )
    guidance = MissionGuidance(reached_m=0.3, final_hold_s=0.0)
    with nx.Sim._from_loop(orch, guidance=guidance) as sim:
        guidance.set_mission(flight.WAYPOINTS[:1])
        sim.run()

    assert guidance.reached == 1


HOVER = (0.0, 0.0, 2.0)  # the hover setpoint, where the flight starts [m]
HOLD_TICKS = 500  # 2 s at the default 4 ms tick
LEVEL = 5.0  # the most the airframe tilts off level over the hold [degrees]
NEAR = 0.2  # the farthest the airframe drifts from the setpoint over the hold [m]


def _hold(path) -> dict:
    """Hold the hover setpoint on the quad at `path` under the PID example's controller for 2 s on the CPU device.
    Returns the airframe's largest tilt [degrees] and distance from the setpoint [m] over the hold, and the error
    the build raised.
    """
    from nexus_uav._src.physics.vehicle import VehicleUsd
    from tests.usd.sensor_vehicle import scenario

    cfg = scenario()
    cfg["physics"]["force_cpu"] = True
    try:
        orch = build_pid_loop(
            cfg, vehicle_usd=VehicleUsd({"usd_path": str(path)}), goal_w=HOVER, spawn=HOVER, max_steps=HOLD_TICKS
        )
        guidance = MissionGuidance(reached_m=0.3, final_hold_s=60.0)
        with nx.Sim._from_loop(orch, guidance=guidance) as sim:
            guidance.set_mission([HOVER])
            sim.run()
            rows = sim["vehicle/body/body"].history_arrays()
    except ValueError as exc:
        return {"tilt": None, "distance": None, "error": str(exc)}
    tilt = max(quad.tilt(row) for row in rows["orientation"])
    distance = float(np.linalg.norm(np.asarray(rows["position"], dtype=np.float64) - np.asarray(HOVER), axis=1).max())
    return {"tilt": tilt, "distance": distance, "error": None}


def test_a_controllers_mixer_allocates_each_rotors_command_from_that_rotors_values(tmp_path):
    """A controller's mixer allocates each rotor's command from that rotor's values.

    Given the fixture quad with rotor 2's `ct` twice the others' under the PID example's controller holding a
    hover setpoint, when the run flies 2 s, then the airframe stays within 5° of level and within 0.2 m of the
    setpoint.
    """
    path = quad.author(tmp_path / "quad.usda", odd_ct=2 * quad.LIFT_CT, channels=(0, 1, 2, 3))

    held = _hold(path)

    level = held["tilt"] is not None and held["tilt"] < LEVEL
    near = held["distance"] is not None and held["distance"] < NEAR
    assert (level, near) == (True, True), held
