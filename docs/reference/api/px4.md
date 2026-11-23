---
description: "API reference for nexus_uav.px4: OffboardClient, the client a script opens on PX4's offboard link, the mission plan types it uploads, and replay, which flies PX4 on a real flight's ULog."
---

# PX4

`nexus_uav.px4` holds PX4's own names: what a script that commands PX4 imports. `OffboardClient` is the
host end of the offboard link of PX4 Software In The Loop (SITL). The run owns that link's address
and names it in its port map, [`sim.ports`](simulation.md), so the script connects the client
itself once the sim has started. `OffboardClient.connect(sim, timeout)` listens on that address and
waits for PX4's heartbeat. PX4 runs on the sim's clock, so the wait steps the sim, and `timeout`
counts sim seconds. When no heartbeat comes, it raises [`RunError`](simulation.md), which names the
port, and so does any wait once the run has lost PX4.

```python
import nexus_uav as nx
from nexus_uav.px4 import OffboardClient

with nx.Sim("astro_max_base", scene="empty") as sim:
    sim.start()
    with OffboardClient.connect(sim, 10.0) as op:  # sim.ports["offboard"]: UDP 14540 on instance 0
        op.takeoff(2.0)
        sim.wait_until(op.at_target, sim_timeout=60.0)
```

A run against the fake PX4 lists no offboard link, so `sim.ports["offboard"]` raises and names the
fake. The mission plan types describe what `upload_mission` hands PX4:

```python
from nexus_uav.px4 import MissionItem, Plan, read_plan
```

::: nexus_uav.px4.OffboardClient
::: nexus_uav.px4.Plan
::: nexus_uav.px4.MissionItem
::: nexus_uav.px4.read_plan

`replay` flies PX4 on the inputs a real flight's ULog holds, see [Replaying a flight](../../guide/replaying-a-flight.md).

::: nexus_uav.px4.replay
