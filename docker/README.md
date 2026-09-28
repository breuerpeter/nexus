# `docker/`

Container images for the Newton project. CI builds `px4-sitl` + publishes it to the
**GitHub Container Registry (GHCR)**, see
[`.github/workflows/docker-images.yml`](../.github/workflows/docker-images.yml), and consumers
**pull it by URL**, no local build needed. The Kit render peer has no image here: it runs NVIDIA's
image as pulled, see [the Kit render peer](#the-kit-render-peer). `mediamtx/` isn't an image: it holds the config for the
compose `mediamtx` service, which pulls `bluenviron/mediamtx`. **RL training**, `nexus-rl`, is *not*
a container. It runs host-side as a separate `uv` project. See the note at the bottom.

## `px4-sitl`

Lean PX4 **Software In The Loop (SITL)** build toolchain. It has only the dependencies to build
Portable Operating System Interface (POSIX) targets for an **external, decoupled simulator**, and no
simulator-specific dependencies. It builds + runs a PX4 checkout against that simulator. The airframe
make-target is the command, and the vehicle is the positional arg.

Nothing starts it by hand. The sim does:

```bash
uv run nexus run --vehicle astro_max_base --control px4-sitl
```

That builds PX4, serves the Hardware In The Loop (HIL) link on :4560, starts this container against
it, and force-removes it on the way out.
[`nexus/_src/vehicle/controllers/px4/sitl.py`](../nexus/_src/vehicle/controllers/px4/sitl.py) defines the
container **once** and is also what runs it, through the docker daemon's Python SDK in
[`nexus/_src/containers.py`](../nexus/_src/containers.py). PX4's console goes to
`~/.cache/nexus/logs/px4-*.log`, next to the run's recording.

PX4 uses host networking, so it reaches port `:4560` of the sim and exposes the Ground Control
Station (GCS) MAVLink on **`:18570`**. Point QGroundControl, or any GCS, there to arm + fly. The
checkout is bind-mounted at its host path, so container and host share one `build/` tree, and
`--user` keeps those build artifacts host-owned. Set `PX4_DIR` if the checkout isn't `~/code/px4`.

You pull the image by URL, no local build needed. `$PX4_IMAGE` names another image outright. To
build or iterate it locally:

```bash
docker build -t ghcr.io/breuerpeter/nexus/px4-sitl:latest docker/px4-sitl
```

## The Kit render peer

A vehicle whose Universal Scene Description (USD) file authors a `Camera` or `OmniLidar` prim
renders it in the **Kit render peer**. The sim starts that container beside PX4 and stops it at the
end of the run. The loop stays on the host, and the peer takes poses over a socket and returns each
frame. The program Kit runs lives in
[`nexus/_src/rendering/kit-peer/`](../nexus/_src/rendering/kit-peer/) and ships in the package, and
[`nexus/_src/rendering/peer.py`](../nexus/_src/rendering/peer.py) runs it through the docker SDK.
No compose service starts it.

This repository builds no Kit image. The container runs NVIDIA's
`nvcr.io/nvidia/isaac-sim:6.0.1` as pulled. The first RTX run on a machine pulls it, with no NGC
login, and later runs reuse it. The peer program mounts read-only at `/nexus-kit`. A scene that
declares a Cesium tileset fetches Cesium for Omniverse into the asset cache, and the container mounts
it at `/cesium-exts`.

The container runs as the host user. It mounts the asset cache and the folder of each local USD it
renders read-only, at their host paths. It mounts `~/.cache/nexus/kit/` for Kit's own caches, which
the run creates as the user first, since docker would create it owned by root. Kit's settings and
logs under `/isaac-sim/kit` stay in the container. Its console goes to
`~/.cache/nexus/logs/console-*.log`.

The Kit-only asset scripts under `scripts/assets/` run in the same image. Each runs from the host
with `uv run python scripts/assets/<script>.py …`, starts the container with its own file and
arguments through `scripts/assets/kit_container.py`, and boots Kit there. The container mounts the
working folder and `$NEXUS_DATA` at their host paths.

## `isaac-lab`: not a container, it runs host-side

The **RL training app** in [`nexus-rl/`](../nexus-rl/) no longer needs a container. It runs
**on the host without Kit** as a **separate `uv` project**. `uv run --project nexus-rl` installs
Isaac Lab on the Newton backend on demand, in its own virtual environment, and trains on the
framework's own Newton and Warp. That works via an editable path dependency on `nexus`, so
training + the standalone deploy share one Newton, for better FR-7 parity. See that app's README.
[`scripts/ci/run_rl_example.sh`](../scripts/ci/run_rl_example.sh) runs the full train → record →
deploy → regression pipeline host-side. Historically this used NVIDIA's `nvcr.io/nvidia/isaac-lab`
image, which it no longer needs.
