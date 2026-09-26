# The Kit image: NVIDIA's Isaac Sim base, Cesium for Omniverse, and the render peer program.
#
# The image cannot be a public package, since it holds NVIDIA's layers, so each machine builds it on
# its first RTX run. nexus tags it with a hash of this folder, the only files the build reads, so a
# host code change never rebuilds it. It holds no nexus code and no nexus dependency: the peer
# program imports only what Kit ships. The base pulls with no NGC login.
FROM nvcr.io/nvidia/isaac-sim:6.0.1

USER root

# Cesium for Omniverse, the live-streamed geolocated globe for --scene cesium. v0.29.0 is the first
# release built for Kit 110 / Isaac Sim 6.0.1. Its two Kit extensions, cesium.omniverse and
# cesium.usd.plugins, unpack to /cesium-exts; the peer adds that folder to the extension search
# path and enables them only for a Cesium scene. Content-pinned by sha256.
ARG CESIUM_OMNIVERSE_VERSION=v0.29.0
ARG CESIUM_OMNIVERSE_SHA256=fe38d6195974620e87d7cdad32bb4b38d5933389523d56e75bbae23bcf4da55f
RUN curl -fsSL -o /tmp/cesium-omniverse.zip \
        "https://github.com/CesiumGS/cesium-omniverse/releases/download/${CESIUM_OMNIVERSE_VERSION}/CesiumGS-cesium-omniverse-linux-x86_64-${CESIUM_OMNIVERSE_VERSION}.zip" \
    && echo "${CESIUM_OMNIVERSE_SHA256}  /tmp/cesium-omniverse.zip" | sha256sum -c - \
    && mkdir -p /cesium-exts \
    && unzip -q /tmp/cesium-omniverse.zip -d /cesium-exts \
    && rm /tmp/cesium-omniverse.zip

# The Cesium USD schemas must register at USD start-up: the schema registry is built once at Kit
# boot, so enabling the extension later registers them too late for prim definitions.
ENV PXR_PLUGINPATH_NAME=/cesium-exts/cesium.usd.plugins/plugins/CesiumUsdSchemas/resources

# The container runs as the host user, with the image's isaac-sim group added to read /isaac-sim,
# so every file it writes into a mounted cache belongs to that user. Kit also writes its settings
# and logs under /isaac-sim/kit, which only isaac-sim owns; these two stay in the container.
RUN mkdir -p /isaac-sim/kit/data /isaac-sim/kit/logs /isaac-sim/kit/cache \
    && chmod 1777 /isaac-sim/kit/data /isaac-sim/kit/logs /isaac-sim/kit/cache

COPY . /nexus-kit/

# The base's check greps Kit's log under the image user's home, which the peer, with its own HOME,
# never writes, so every healthy peer read as unhealthy. The render link says whether the peer lives.
HEALTHCHECK NONE

USER isaac-sim
ENTRYPOINT ["/isaac-sim/python.sh"]
CMD ["/nexus-kit/serve.py"]
