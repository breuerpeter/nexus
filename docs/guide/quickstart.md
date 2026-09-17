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

Run the command-line tool with the default vehicle and controller, `astro-max` + `px4-sitl`:

```bash
uv run nexus run
```

This builds PX4 Software In The Loop (SITL) and starts the physics simulation with the Rerun
recording server on gRPC port 9876. Then it launches PX4 in a container that connects back over
TCP port 4560. One command, no second terminal. It needs a PX4-Autopilot checkout at `$PX4_DIR`,
default `~/code/px4`. Pick a different vehicle with `--vehicle`. PX4 is the first-class
controller, and the other controllers are self-contained examples.

## Run an example

Launch the [Rerun](https://rerun.io) viewer, then fly the sampling Model Predictive Control (MPC)
obstacle slalom and watch it live:

```bash
uv pip install nexus-sim
uv run -m nexus.examples sampling_mpc
uv run rerun ~/.cache/nexus/logs/*.rrd
```

List every example with `uv run -m nexus.examples --list`. Each example is zero-arg, its
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
