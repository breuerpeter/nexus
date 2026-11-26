---
description: "Replay a real flight's ULog into PX4 Software In The Loop (SITL): the same sticks, commands and parameters, with one change of your own."
---

# Replaying a flight

A replay flies PX4 Software In The Loop (SITL) on the inputs a real flight had. It reads them from the flight's ULog, and its run keeps PX4's own ULog. Open the two logs side by side to see how the flight would have gone with one parameter changed.

```python
from nexus_uav.px4 import replay

artifacts = replay(
    "flight.ulg",
    "astro_max_base",
    scene="empty",
    params={"MPC_XY_VEL_MAX": 5.0},
)
print(artifacts["ulog"])  # PX4's own log of the replayed flight
```

Install the extra first: `uv add "nexus-uav[replay]"`. It brings `pyulog`, which reads the log, and `cyclonedds`, which writes the inputs.

## How it differs from PX4's own replay

PX4 ships a replay of its own, which publishes a log's topics back into a PX4 with no vehicle behind it. The logged sensor rows are fixed, so they don't react to what PX4 does with them. That suits a question about the estimator: the same sensor rows, another estimator parameter. It can't answer how the flight would have gone, since a changed controller moves no vehicle.

This replay sends only what reached the vehicle from outside it, and the sim's vehicle answers PX4. So a changed parameter changes the flight, and the replay's log shows it.

Two more facts keep PX4's replay out of a sim run. It sets PX4's lockstep clock itself, so no simulator can step PX4 beside it. And it needs a PX4 build of its own, where this replay flies the log's own commit unchanged.

## What the replay sends

The ULog is the only source of what the replay sends. Each input goes back into PX4 as the topic the log holds, with its logged content and at its logged time, through PX4's bridge to the Data Distribution Service (DDS). The replay changes nothing in PX4.

| The log's topic | What it carries | The bridge input |
|---|---|---|
| `manual_control_setpoint` | the sticks | `manual_control_input` |
| `vehicle_command`, the external rows | commands from the ground station and the companion computer | `vehicle_command` |
| `telemetry_status` | the status of each link | `telemetry_status` |

A log holds a subsample of the sticks and drops rows. So the replay holds the last stick sample across a gap while the next row is valid. Without the hold, PX4 would see a stick loss the flight never had.

## What the replay leaves out

The bridge lists no input for the switches, the gimbal command or the air traffic, so the replay sends none of them. A flight that changed mode by a switch replays without that change.

The log doesn't hold the stored mission, the correction data of a base station, or the wind. A replay that starts at takeoff needs none of the three. The sim's wind and payload aren't the flight's, so the two logs differ where those matter.

## The PX4 it flies

The replay flies the commit the log names in `ver_sw`, from the repository of the vehicle's PX4 pin. That keeps PX4's start state the flight's: a PX4 that boots in another mode than the log's flies another flight. Name another PX4 with `px4`, a `repo@sha` or the path of a checkout:

```python
replay("flight.ulg", "astro_max_base", scene="empty", px4="PX4/PX4-Autopilot@2de9ea3e0823391b648204cc3921bdd490442e46")
```

A commit the pin's repository lacks fails before any peer starts, and the error names the commit. `NEXUS_PX4_SITL_REF` in the environment names the PX4 for every run, the replay too.

## The parameters PX4 boots on

PX4 boots on the log's parameters, then on yours, `params`. The replay skips a parameter only when it denies a SITL flight, such as `COM_POWER_COUNT`, which asks for a second power source. It sets `UXRCE_DDS_SYNCT` to 0, so PX4 takes each sample's time as sent when the run is faster than real time.

## What the run starts

The replay authors an override layer over the vehicle. The layer sets `nexus:params` on the vehicle's PX4 SITL peer, and declares the Micro XRCE-DDS agent peer, `NexusXrceAgentAPI`, the bridge's other end. The agent listens on the User Datagram Protocol (UDP) port 8888 plus the run's instance, and PX4 dials the same port, so two replays on one machine don't meet. The first run builds the agent's image, which takes a few minutes.
