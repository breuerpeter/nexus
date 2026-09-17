#!/usr/bin/env bash
# Provision a PX4 SITL checkout for the ephemeral GPU runner (or any fresh box). The checkout only:
# building it belongs to whoever runs PX4, which does it from the ONE container definition in
# nexus/_src/vehicle/controllers/px4/sitl.py (`build_px4_sitl()`).
#
# The pin lives in scripts/ci/px4.ref line 1 (`repo@sha`) — the PX4 tree this checkout flies.
# px4-bump.yml keeps it fresh via reviewed, gpu-labeled PRs that fly the flight gate before
# merging. PX4_REPO/PX4_REF/PX4_DIR override the file, for a local experiment or another
# checkout; a clone that needs authentication takes $PX4_CLONE_TOKEN on the fetch command line
# alone, and the tracked pin clones tokenless.
# The airframe difference self-handles: a tree whose none_astro_max carries the full mag config
# no-ops the interim patch below; one without it gets the patch until #11 fixes the sim's mag.
# Idempotent: the clone is skipped when $PX4_DIR already exists (e.g. restored from the
# actions cache keyed on px4.ref).
#
#   bash scripts/ci/provision_px4.sh
set -euo pipefail

# The ephemeral CI runner's job env has no $HOME (the same quirk gpu-runner.yml works around for
# setup-uv) — derive it from the passwd db so the PX4_DIR default resolves under `set -u`.
export HOME="${HOME:-$(getent passwd "$(id -u)" | cut -d: -f6)}"

REF_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/px4.ref"
IFS=@ read -r REF_REPO REF_SHA < "$REF_FILE"
PX4_REPO="${PX4_REPO:-$REF_REPO}"
PX4_REF="${PX4_REF:-$REF_SHA}"
PX4_DIR="${PX4_DIR:-$HOME/code/px4}"

if [ ! -e "$PX4_DIR/Makefile" ]; then
  echo "cloning $PX4_REPO @ $PX4_REF -> $PX4_DIR"
  mkdir -p "$PX4_DIR"
  git -C "$PX4_DIR" init -q
  # The token rides ONLY the fetch command line — never a stored remote:
  # the checkout is cached by CI, and a token in .git/config would persist into the cache.
  FETCH_URL="https://github.com/$PX4_REPO.git"
  if [ -n "${PX4_CLONE_TOKEN:-}" ]; then
    FETCH_URL="https://x-access-token:${PX4_CLONE_TOKEN}@github.com/$PX4_REPO.git"
  fi
  git -C "$PX4_DIR" fetch -q --depth 1 "$FETCH_URL" "$PX4_REF"
  git -C "$PX4_DIR" checkout -q FETCH_HEAD
  git -C "$PX4_DIR" submodule update --quiet --init --recursive
fi

# INTERIM (remove when the branch carries it and px4.ref bumps): the 80000_none_astro_max
# airframe lacks EKF2_MAG_TYPE 6 (heading from mag at init only), so the EKF's continuous mag
# fusion faults in flight ("Compass 0 fault") and trips the px4_warnings gate. Verified fix —
# belongs in PX4 PR #27706.
AIRFRAME="$PX4_DIR/ROMFS/px4fmu_common/init.d-posix/airframes/80000_none_astro_max"
if [ -e "$AIRFRAME" ] && ! grep -q EKF2_MAG_TYPE "$AIRFRAME"; then
  printf '\nparam set-default EKF2_MAG_TYPE 6\n' >> "$AIRFRAME"
  echo "patched $AIRFRAME with EKF2_MAG_TYPE 6 (interim — PX4 PR #27706)"
fi

echo "PX4 checkout at $PX4_DIR ($(git -C "$PX4_DIR" rev-parse --short HEAD))"
