---
description: "How to log from examples and user code: the framework logger, which routes to the console and the Rerun recording."
---

# Logging

Log from examples and user code through the framework logger, `na.logger`, which is the standard-library logger named `newton`:

```python
import nexus as na

na.logger.info("reached the waypoint")
na.logger.warning("flight degraded")
```

It always writes to the **console**, on stderr. When a recording is active, through `--log` or `--view`,
or `Sim(log=…)` or `Sim(view=…)`, it **also** tees every record into the recording's `logs/sim` panel
as a Rerun `TextLog`. So a result shows up in the terminal, including in headless CI where no recording
exists, *and* in the `.rrd` for review. No bare `print` needed.

The framework itself logs the same way, so example output sits alongside the framework's own
`[newton/…]` lines. Components log their own *quantities*: the scene, the flown-path
`physics/trajectory`, the operator's `operator/reference`/waypoints, the `controller/mpc_horizon`, each
through its own component log step, which the central `Logger` fans out only when recording.
