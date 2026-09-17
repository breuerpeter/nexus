# Isaac Sim runtime image for the decoupled Newton drone sim (architecture.md §11/§12).
#
# Hosts the in-process Isaac Sim Kit app as a pure RENDERER around the SAME core Orchestrator +
# components as the standalone runtime — including the SAME NewtonPhysics (our ModelBuilder + our
# solvers; Kit never simulates, Isaac's physics backend is unused). PX4 SITL dials into the HIL
# server this serves on TCP:4560 — same topology as the standalone runtime — so a flight is
# `docker compose up isaacsim` + the px4-sitl service. GPU-enabled (RTX, NVIDIA container runtime).
#
# Only EXTERNAL Python deps are baked in; the nexus package itself is bind-mounted at run
# time (the compose service) and added to PYTHONPATH by the entrypoint — mirroring px4-sitl's
# bind-mount workflow (edit code, re-run, no rebuild).
FROM nvcr.io/nvidia/isaac-sim:6.0.1

LABEL org.opencontainers.image.description="Isaac Sim runtime (Newton physics backend) for the decoupled PX4 HIL drone sim; serves the :4560 HIL link."

# The base image runs as a non-root user; build (and run) as root — Isaac Sim writes to its own
# /isaac-sim dirs, and the runtime only reads the bind-mounted repo + writes dedicated cache mounts.
USER root

# ffmpeg for the FPV video source (A2 Plan 3): the in-process renderer pipes RTX frames to
# `ffmpeg -c:v h264_nvenc …` → RTSP → MediaMTX → WHEP. Ubuntu's ffmpeg ships the h264_nvenc
# encoder; the NVIDIA container runtime provides the encode driver libs at run time. NVENC is a
# separate HW block (no CUDA-core contention with Warp/MuJoCo physics).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python deps are NOT baked: the entrypoint syncs everything from the bind-mounted repo at
# container start (ONE source of truth, hash-stamped + a persistent pip-cache mount, so unchanged
# inputs add zero boot time), in two tiers:
#   1. general deps from pyproject (skipping the Kit-bundled newton/warp/usd names — those go via
#      tier 2; usd/pxr always stays Kit's, it is ABI-bound to the platform);
#   2. PHYSICS PINS from uv.lock (newton, warp-lang, newton-usd-schemas, mujoco, mujoco-warp) into
#      $NEXUS_PINS_DIR — the runtime applies them around the Kit boot (apply_pins_pre/post_boot:
#      pre-boot warp import + prebundle-extension exclusion beat Kit's finder-level bundling), so
#      the container runs the SAME newton/warp as the host. No version skew: the CPU PID parity
#      gate is bit-identical on the pinned stack (scripts/probes/pid_parity_probe.py). Without
#      pins (probes run outside the entrypoint, stale caches) Kit's bundled builds still fly.

# Cesium for Omniverse — the live-streamed geolocated globe for the FPV world (--scene cesium).
# v0.29.0 (2026-07-01) is the first upstream release built for Kit 110 / Isaac Sim 6.0.1, which
# retires docs/assets/s4b-photoreal/CESIUM_KIT110_BLOCKER.md (the v0.28 USD/absl ABI mismatches).
# The two Kit extensions (cesium.omniverse + cesium.usd.plugins) are unpacked to /cesium-exts;
# the runtime adds that folder to the extension search path and enables them only when the launch
# selects a cesium world (RtxFrame._setup_cesium). Content-pinned by sha256.
ARG CESIUM_OMNIVERSE_VERSION=v0.29.0
ARG CESIUM_OMNIVERSE_SHA256=fe38d6195974620e87d7cdad32bb4b38d5933389523d56e75bbae23bcf4da55f
RUN curl -fsSL -o /tmp/cesium-omniverse.zip \
        "https://github.com/CesiumGS/cesium-omniverse/releases/download/${CESIUM_OMNIVERSE_VERSION}/CesiumGS-cesium-omniverse-linux-x86_64-${CESIUM_OMNIVERSE_VERSION}.zip" \
    && echo "${CESIUM_OMNIVERSE_SHA256}  /tmp/cesium-omniverse.zip" | sha256sum -c - \
    && mkdir -p /cesium-exts \
    && unzip -q /tmp/cesium-omniverse.zip -d /cesium-exts \
    && rm /tmp/cesium-omniverse.zip

COPY entrypoint.sh /usr/local/bin/newton-isaac-entrypoint.sh
RUN chmod +x /usr/local/bin/newton-isaac-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/newton-isaac-entrypoint.sh"]
CMD ["run", "--runtime", "isaacsim", "--vehicle", "astro-max", "--control", "px4-sitl"]
