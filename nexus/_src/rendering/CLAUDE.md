The Kit render peer: `peer.py` builds the Kit image and runs its container, and `link.py` is the
renderer the loop drives and the link the RTX sensors ride.

- `kit-peer/` is the program the container runs under Kit's Python, shipped as package data. No
  module imports it, and it imports no nexus module, so the `no-kit` contract in `.importlinter`
  holds for every tier.
- A module in `kit-peer/` runs as a top-level module in the image, so its name must not shadow a
  Kit package: the Cesium handler is `cesium_globe.py` because Cesium's extension is the `cesium`
  package.
- Any change under `kit-peer/` changes `image_tag()`, a hash of that folder minus what
  `.dockerignore` leaves out, so the next RTX run rebuilds the image. A change anywhere else never
  does.
- The wire format lives twice, in `link.py` and `kit-peer/link.py`: change both.
  `tests/rendering/test_link.py` drives the host end against a stand-in peer that frames through
  `kit-peer/link.py`.
- The container runs as the host user with the image's `isaac-sim` group, gid `1234`, added, since
  only that group can read `/isaac-sim`. Kit's `HOME` is `~/.cache/nexus/kit/home` and its shader
  cache `~/.cache/nexus/kit/cache`.
- The peer's console is `~/.cache/nexus/logs/console-*.log`: its own lines start `[kit-peer]`, and
  every 240 frames one of them splits a frame's time into the wait for the host, the render and
  the send.
