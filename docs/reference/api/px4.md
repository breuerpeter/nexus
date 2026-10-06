---
description: "API reference for nexus.px4: OffboardClient, the client a script opens on PX4's offboard link, and the mission plan types it uploads."
---

# PX4

`nexus.px4` holds PX4's own names: what a script that commands PX4 imports. `OffboardClient` is the
host end of the offboard link of PX4 Software In The Loop (SITL). The run owns that link's address
and names it in its port map, [`sim.ports`](simulation.md), so the script opens the client itself
once the sim has started. Entering the client returns at once. PX4 runs on the sim's clock, so the
wait for its heartbeat steps the sim.

```python
import nexus as na
from nexus.px4 import OffboardClient

with na.Sim("astro_max_base", scene="empty") as sim:
    sim.start()
    link = sim.ports["offboard"]  # {"protocol": "udp", "port": 14540, "system_id": 1} on instance 0
    with OffboardClient(f"udpin:0.0.0.0:{link['port']}", system_id=link["system_id"]) as op:
        sim.wait_until(lambda: op.connected, sim_timeout=10.0)
        op.takeoff(2.0)
        sim.wait_until(op.at_target, sim_timeout=60.0)
```

A run against the fake PX4 lists no offboard link, so `sim.ports["offboard"]` raises and names the
fake. The mission plan types describe what `upload_mission` hands PX4:

```python
from nexus.px4 import MissionItem, Plan, read_plan
```

::: nexus.px4.OffboardClient
::: nexus.px4.Plan
::: nexus.px4.MissionItem
::: nexus.px4.read_plan
