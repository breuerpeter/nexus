# Lean PX4 SITL build image: only the dependencies needed to build POSIX (SITL) targets for use
# with an external (decoupled) simulator. No simulator-specific dependencies. Consumers pull this
# image from GHCR and build PX4 SITL against a mounted px4 checkout
# (e.g. `docker run ... make px4_sitl none_astro_max`).
FROM ubuntu:24.04

LABEL org.opencontainers.image.description="Lean PX4 SITL POSIX build image (external/decoupled simulator); no simulator-specific dependencies."

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        g++ \
        git \
        make \
        ninja-build \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# PX4 build-time Python deps. `future` is required by pymavlink's C generator; the rest are the
# minimal POSIX-build set. (No numpy/packaging/toml/etc. — not needed for a SITL build.)
RUN uv pip install --system --break-system-packages --no-cache \
    "empy>=3.3,<4" \
    "jinja2>=2.8" \
    future \
    jsonschema \
    kconfiglib \
    pyros-genmsg \
    pyyaml

# No git safe.directory shim is needed: the compose `px4-sitl` service runs the container as the
# host UID (matching the bind-mounted px4 checkout's owner), so git sees the repo as its own and
# build artifacts land host-owned (no root-owned build/ tree to clean up).
