# `docker/`

Container images for the Newton project, built + published to the
**GitHub Container Registry (GHCR)** by CI, see
[`.github/workflows/docker-images.yml`](../.github/workflows/docker-images.yml). Consumers **pull by
URL**, no local build needed. One subdirectory per image, `px4-sitl` and `isaacsim` so far. Add more
over time. `mediamtx/` isn't an image: it holds the config for the compose `mediamtx` service, which
pulls `bluenviron/mediamtx`. **RL training**, `nexus-rl`, is *not* a container. It runs
host-side as a separate `uv` project. See the note at the bottom.

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

PX4 uses host networking, so it reaches port `:4560` of the runtime and exposes the Ground Control
Station (GCS) MAVLink on **`:18570`**. Point QGroundControl, or any GCS, there to arm + fly. The
checkout is bind-mounted at its host path, so container and host share one `build/` tree, and
`--user` keeps those build artifacts host-owned. Set `PX4_DIR` if the checkout isn't `~/code/px4`.

You pull the image by URL, no local build needed. `$PX4_IMAGE` names another image outright. To
build or iterate it locally:

```bash
docker build -t ghcr.io/breuerpeter/nexus/px4-sitl:latest docker/px4-sitl
```

## `isaacsim`

The **in-process Isaac Sim runtime**, on the GPU with **RTX** and the NVIDIA container runtime: the
photorealistic Isaac half of the decoupled stack. It hosts the *same* core `Orchestrator` + components
as the standalone runtime, including the same `NewtonPhysics`. It doesn't use Isaac's physics
backend. Kit only renders, see `nexus/_src/runtimes/isaacsim/runtime.py`. It serves the HIL link
on `:4560` exactly as the standalone runtime does, so PX4 SITL dials in identically. The image bakes
in only external Python dependencies, see `docker/isaacsim/Dockerfile` and its `FROM
nvcr.io/nvidia/isaac-sim:6.0.1`. `nexus` is bind-mounted at run time, so code edits need no
rebuild, which mirrors `px4-sitl`.

### Host paths

`NEXUS_DIR` is this checkout, bind-mounted **at the same path inside the container** and added
to `PYTHONPATH`. **You don't set it**: `nexus run` and `nexus script` derive it from the
checkout they run out of, which is also where they find this file. The code is in
`nexus/_src/cli/detect.py`.

Three mounts need a source that **already exists**: the two checkouts and the daemon socket. Each
declares `create_host_path: false`, so docker names a wrong path instead of creating it root-owned
and leaving the container with an empty directory. Cache mounts keep the short syntax, because
docker should create those on first use.

A full Isaac Sim flight:

```bash
# 1. boot Kit, serve the HIL link on :4560, and start PX4 SITL against it (the controller does that
#    from in here, over the daemon socket this service mounts):
docker compose -f docker/docker-compose.yml up isaacsim
# 2. point a GCS (QGC, or a script against `Px4Offboard` on :14540) at the link → arm + takeoff
```

Any Kit-only Python runs through the command-line tool. `uv run nexus script <path> [args…]`
auto-launches this service and runs the target under the booted Kit app, with the dependency sync of
the entrypoint and the uv.lock physics pins applied. Nothing overrides the entrypoint.

Pulled by URL from `ghcr.io/breuerpeter/nexus/isaacsim-runtime:latest`. To build or
iterate locally:

```bash
docker compose -f docker/docker-compose.yml build isaacsim
```

## `isaac-lab`: not a container, it runs host-side

The **RL training app** in [`nexus-rl/`](../nexus-rl/) no longer needs a container. It runs
**on the host without Kit** as a **separate `uv` project**. `uv run --project nexus-rl` installs
Isaac Lab on the Newton backend on demand, in its own virtual environment, and trains on the
framework's own Newton and Warp. That works via an editable path dependency on `nexus`, so
training + the standalone deploy share one Newton, for better FR-7 parity. See that app's README.
[`scripts/ci/run_rl_example.sh`](../scripts/ci/run_rl_example.sh) runs the full train → record →
deploy → regression pipeline host-side. Historically this used NVIDIA's `nvcr.io/nvidia/isaac-lab`
image, which it no longer needs.
