#!/bin/bash
# Entrypoint for the Isaac Sim runtime service: make the bind-mounted nexus package importable
# under Kit's Python and run the CLI. Mirrors px4-sitl's "bind-mount the checkout" workflow — code
# edits are picked up without an image rebuild.
set -e

NEXUS_DIR="${NEXUS_DIR:-/workspace}"
if [ ! -d "${NEXUS_DIR}/nexus" ]; then
    echo "newton-isaac: nexus not found at NEXUS_DIR=${NEXUS_DIR} (bind-mount the newton repo there)" >&2
    exit 1
fi

# The single `nexus` package lives at the repo root; put the root on PYTHONPATH so Kit's
# Python imports the workspace code. (The single-package layout means `nexus._src.vehicle.actuators`
# can't collide with NVIDIA Newton's bundled top-level `newton_actuators` — no rebinding needed.)
export PYTHONPATH="${NEXUS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Cesium USD schemas must be registered at USD static init (PXR_PLUGINPATH_NAME): the
# UsdSchemaRegistry is built once at Kit boot, so a runtime extension-enable registers the plugin
# too late for prim DEFINITIONS ("Empty typeName" on the cesium:* attrs — the blocker doc's
# symptom). The extension itself (tile engine) still only loads when a launch selects a cesium
# world (RtxFrame._setup_cesium).
CESIUM_EXTS="${NEXUS_CESIUM_EXTS:-/cesium-exts}"
CESIUM_SCHEMAS="${CESIUM_EXTS}/cesium.usd.plugins/plugins/CesiumUsdSchemas/resources"
if [ -d "${CESIUM_SCHEMAS}" ]; then
    export PXR_PLUGINPATH_NAME="${CESIUM_SCHEMAS}${PXR_PLUGINPATH_NAME:+:${PXR_PLUGINPATH_NAME}}"
fi

# Isaac Sim EULA / privacy (non-interactive).
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

# Dependency sync — ONE source of truth (the repo's pyproject), no Dockerfile duplicate list.
# Installs the project's external deps into Kit's Python at container start, SKIPPING everything
# Kit bundles (newton/warp/usd — pulling the workspace pins would shadow the builds Kit's own
# extensions are compiled against) and the heavy optional extras. The stamp lives in the SAME
# ephemeral layer as the install target (Kit's site-packages die with the container, so a
# persistent stamp would skip the sync in every later container — the deps vanish while the stamp
# says they're there); the mounted pip cache still makes the re-sync near-instant.
DEPS_STAMP="/isaac-sim/.newton-deps.stamp"
PYPROJECT_HASH=$(sha256sum "${NEXUS_DIR}/pyproject.toml" | cut -d' ' -f1)
if [ "$(cat "${DEPS_STAMP}" 2>/dev/null)" != "${PYPROJECT_HASH}" ]; then
    echo "newton-isaac: syncing deps from pyproject (${PYPROJECT_HASH:0:12}…)"
    /isaac-sim/python.sh - "$NEXUS_DIR" <<'PY' > /tmp/newton-reqs.txt
import sys
import tomllib

deps = tomllib.load(open(f"{sys.argv[1]}/pyproject.toml", "rb"))["project"]["dependencies"]
KIT_BUNDLED = ("newton", "warp-lang", "usd-core", "newton-usd-schemas")
for d in deps:
    name = d.split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip()
    if name not in KIT_BUNDLED:
        print(d)
PY
    /isaac-sim/python.sh -m pip install -q -r /tmp/newton-reqs.txt
    mkdir -p "$(dirname "${DEPS_STAMP}")" && echo "${PYPROJECT_HASH}" > "${DEPS_STAMP}"
fi

# Physics pins — the WORKSPACE newton/warp stack (uv.lock's resolved versions, ONE source of
# truth) into a dedicated dir the runtime prepends at boot, so the container runs the SAME
# newton/warp as the host (no version skew, no Kit-bundle special cases). Installed --no-deps
# (everything else comes from Kit / the dep sync above); stamped on the uv.lock hash.
export NEXUS_PINS_DIR="/root/.cache/pip/newton-pins"
PINS_STAMP="${NEXUS_PINS_DIR}.stamp"
LOCK_HASH=$(sha256sum "${NEXUS_DIR}/uv.lock" | cut -d' ' -f1)
if [ "$(cat "${PINS_STAMP}" 2>/dev/null)" != "${LOCK_HASH}" ]; then
    PINS=$(/isaac-sim/python.sh - "$NEXUS_DIR" <<'PY'
import sys
import tomllib

lock = tomllib.load(open(f"{sys.argv[1]}/uv.lock", "rb"))
WANT = ("newton", "warp-lang", "newton-usd-schemas", "mujoco", "mujoco-warp")
for p in lock["package"]:
    if p["name"] in WANT:
        print(f"{p['name']}=={p['version']}")
PY
)
    echo "newton-isaac: syncing physics pins from uv.lock: $(echo $PINS | tr '\n' ' ')"
    rm -rf "${NEXUS_PINS_DIR}"
    /isaac-sim/python.sh -m pip install -q --no-deps --target "${NEXUS_PINS_DIR}" ${PINS}
    echo "${LOCK_HASH}" > "${PINS_STAMP}"
fi

exec /isaac-sim/python.sh -m nexus._src.cli.main "$@"
