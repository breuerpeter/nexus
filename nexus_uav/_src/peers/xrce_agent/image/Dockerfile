# The Micro XRCE-DDS agent, built from the commit the peer's pin names. The runner, ../runner.py, passes the
# pin's repository and commit as the two build arguments, so the image's tag changes with the pin.
FROM ubuntu:24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG REPO
ARG SHA

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        g++ \
        git \
        make \
    && rm -rf /var/lib/apt/lists/*

# The agent's own build fetches the libraries it links, Fast DDS among them, at the versions its commit names.
RUN git init -q /agent && \
    git -C /agent fetch -q --depth 1 "https://github.com/${REPO}.git" "${SHA}" && \
    git -C /agent checkout -q FETCH_HEAD && \
    cmake -S /agent -B /agent/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/agent && \
    cmake --build /agent/build -j "$(nproc)" && \
    cmake --install /agent/build

FROM ubuntu:24.04

LABEL org.opencontainers.image.description="The Micro XRCE-DDS agent, the bridge between PX4's topics and DDS."

COPY --from=build /opt/agent /opt/agent
ENV LD_LIBRARY_PATH=/opt/agent/lib
ENV PATH=/opt/agent/bin:$PATH
