#!/usr/bin/env bash
# One-time setup for the acados NMPC example (nexus/examples/controllers/acados_nmpc/waypoint_tracking.py).
#
# acados is not a plain pip dependency: it code-generates and compiles a C solver per problem, so it
# needs (1) the acados C library built from source, (2) the Tera template renderer binary, and (3) the
# matching `acados_template` Python package. This script provisions all three. It is idempotent — the
# expensive C build is skipped if the library already exists.
#
#   bash scripts/setup_acados.sh
#   uv run --extra acados python nexus/examples/controllers/acados_nmpc/waypoint_tracking.py --record
#
# Override the install location with ACADOS_SOURCE_DIR (default: ~/.cache/nexus/acados). The example
# self-configures to the same default, so no environment variables are needed afterwards.
set -euo pipefail

# The ephemeral CI runner's job env has no $HOME — derive it so the default resolves under `set -u`.
export HOME="${HOME:-$(getent passwd "$(id -u)" | cut -d: -f6)}"

ACADOS_DIR="${ACADOS_SOURCE_DIR:-$HOME/.cache/nexus/acados}"
TERA_VERSION="0.2.0"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "acados directory: $ACADOS_DIR"

# 1. acados C library. qpOASES's vendored C99 does not compile under gcc >= 14, so we disable it;
#    HPIPM is acados' default QP solver, and builds cleanly.
if [ ! -f "$ACADOS_DIR/lib/libacados.so" ]; then
  command -v cmake >/dev/null || { echo "cmake not found — install it (e.g. 'pipx install cmake' or 'apt install cmake')." >&2; exit 1; }
  [ -d "$ACADOS_DIR/.git" ] || git clone https://github.com/acados/acados.git "$ACADOS_DIR"
  git -C "$ACADOS_DIR" submodule update --init --recursive
  cmake -S "$ACADOS_DIR" -B "$ACADOS_DIR/build" \
    -DACADOS_WITH_QPOASES=OFF -DACADOS_INSTALL_DIR="$ACADOS_DIR" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$ACADOS_DIR/build" --target install -j
else
  echo "acados C library already built — skipping."
fi

# 2. Tera template renderer (acados' codegen step needs it; it is not part of the C build).
if [ ! -x "$ACADOS_DIR/bin/t_renderer" ]; then
  mkdir -p "$ACADOS_DIR/bin"
  curl -sSL "https://github.com/acados/tera_renderer/releases/download/v${TERA_VERSION}/t_renderer-v${TERA_VERSION}-linux-amd64" \
    -o "$ACADOS_DIR/bin/t_renderer"
  chmod +x "$ACADOS_DIR/bin/t_renderer"
fi

# 3. Python interface, handled by uv (no imperative pip install). acados_template is NOT on PyPI — acados
#    ships it only inside the built source tree, version-locked to libacados.so — so pyproject declares it
#    as an editable path source at the repo-relative `.acados/` (with static dependency-metadata so the base
#    `uv sync` doesn't need it). Point `.acados` at the build dir (gitignored symlink), then `uv sync` it in.
ln -sfn "$ACADOS_DIR" "$REPO/.acados"
( cd "$REPO" && uv sync --extra acados )

echo
echo "acados ready. Run the example with:"
echo "    uv run --extra acados python nexus/examples/controllers/acados_nmpc/waypoint_tracking.py --record"
