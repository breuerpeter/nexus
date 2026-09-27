#!/usr/bin/env bash
# The px4-sitl image on a GPU box: `load` puts a saved image back into the daemon, `save` writes the
# image the run built to the file the box's cache entry keeps. The image builds on the box the way a
# user's first run does, from the Dockerfile the package ships, tagged by that folder's hash; this
# script only spares the box the ~25 s build and the ~185 MB download on a cache hit. The cache entry
# keys on that Dockerfile, so a saved image always carries the tag the run expects.
#
#   bash scripts/ci/px4_image_cache.sh load|save
set -euo pipefail

TAR="${PX4_SITL_TAR:-$HOME/.cache/nexus/ci/px4-sitl.tar}"
IMAGE=nexus-px4-sitl

case "${1:-}" in
  load)
    if [ -f "$TAR" ]; then
      docker load -i "$TAR"
    else
      echo "no saved $IMAGE image at $TAR: the run builds it"
    fi
    ;;
  save)
    mkdir -p "$(dirname "$TAR")"
    if docker image inspect "$(docker images "$IMAGE" --format '{{.Repository}}:{{.Tag}}' | head -n1)" >/dev/null 2>&1; then
      docker save "$(docker images "$IMAGE" --format '{{.Repository}}:{{.Tag}}' | head -n1)" -o "$TAR"
      echo "saved $IMAGE to $TAR"
    else
      echo "no $IMAGE image to save"
    fi
    ;;
  *)
    echo "usage: $0 load|save" >&2
    exit 2
    ;;
esac
