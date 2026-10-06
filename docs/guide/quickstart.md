---
description: "Install nexus and run your first GPU-accelerated drone simulation in a couple of minutes."
---

# Quickstart

Install nexus and run your first simulation in a couple of minutes.

## Requirements

nexus requires **Python 3.12**, **Linux**, and a **CUDA-capable GPU**.

## Install

From PyPI:

```bash
pip install nexus-sim
```

Optional extras: `policy` for PyTorch, `examples` for jerk-limited trajectories, `upload` for S3
asset upload, or `all`.

For development, use [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/breuerpeter/nexus.git
cd nexus
uv sync                 # install the framework
```

## Run the simulator

Run the command-line tool with a vehicle and a scene. `astro_max_base` declares PX4 as its controller, and `empty` is flat ground:

```bash
uv run nexus run --vehicle astro_max_base --scene empty
```

This starts PX4 Software In The Loop (SITL) in a container and the physics simulation, and PX4
connects back over TCP port 4560. Add `--view` to serve the Rerun recording on gRPC port 9876,
or `--log` to write it to an `.rrd` file. One command,
no second terminal. It needs Docker and nothing else: the first run fetches the PX4-Autopilot
commit the framework pins into `~/.cache/nexus/px4/` and builds it there, which takes minutes once,
and builds the small `px4-sitl` image. Set `PX4_DIR` to fly a PX4 checkout of your own instead.
Pick a different vehicle with `--vehicle`. PX4 is the first-class controller, and the other
controllers are self-contained examples.

## Run an example

Launch the [Rerun](https://rerun.io) viewer, then fly the sampling Model Predictive Control (MPC)
obstacle slalom and watch it live:

```bash
uv pip install nexus-sim
uv run -m nexus_sim.examples sampling_mpc
uv run rerun ~/.cache/nexus/logs/*.rrd
```

List every example with `uv run -m nexus_sim.examples --list`. Each example is zero-arg, its
configuration lives in the script, and it records an interactive `.rrd` of its flight.

## Next steps

<div class="grid cards" markdown>

-   **Fly a full SITL flight**, nexus + PX4 SITL + QGroundControl, with the
    port map and gotchas.

    [:octicons-arrow-right-24: Running](running.md)

-   **Work through the examples**: differentiable optimization, MPC, and RL.

    [:octicons-arrow-right-24: Examples](../examples/index.md)

-   **Look up the API**: `Sim`, `Orchestrator`, and the rest of the public surface.

    [:octicons-arrow-right-24: API reference](../reference/index.md)

</div>
