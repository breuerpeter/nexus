The Kit render peer: `runner.py` pulls NVIDIA's Isaac Sim image and runs its container, and
`peer-src/` is the program the container runs. The render link's host end is
`nexus/_src/rendering/link.py`.

- `peer-src/` runs under Kit's Python, shipped as package data. No module imports it, and it imports
  no nexus module, so the `no-kit` contract in `.importlinter` holds for every tier.
- A module in `peer-src/` runs as a top-level module in the image, so its name must not shadow a
  Kit package: the Cesium handler is `cesium_globe.py` because Cesium's extension is the `cesium`
  package.
- The container runs `nvcr.io/nvidia/isaac-sim` as pulled, `IMAGE` in `runner.py`. `peer-src/`
  mounts read-only at `/nexus-kit`, so an edit there takes effect on the next RTX run. Anything the
  peer needs goes in as a run option or a mount, never as a derived image, so an Isaac upgrade
  stays a change to `IMAGE`.
- Cesium for Omniverse is a content-addressed download, `CESIUM` in `runner.py`. The host fetches it
  into the asset cache only when a file the run renders declares a `CesiumTilesetPrim`, and mounts
  it at `/cesium-exts`, as `scripts/assets/kit_container.py` does for the Cesium author script.
- `scripts/assets/kit_container.py` starts the same image for the Kit-only asset scripts and
  imports `ensure_image`, `_run_options`, `_cesium_mount`, `LABEL` and `client` from `runner.py`:
  renaming or moving one of them changes that file too.
- The render link's framing and its message set live once, in `peer-src/link.py`. The peer imports
  it as a sibling module. The host end runs the same file by path and takes `send`, `recv` and the
  message names from it. `tests/rendering/test_link.py` drives the host end against a stand-in
  peer that speaks through it too.
- The container runs as the host user with the image's `isaac-sim` group, gid `1234`, added, since
  only that group can read `/isaac-sim`. Kit's `HOME` is `~/.cache/nexus/kit/home` and its shader
  cache `~/.cache/nexus/kit/cache`.
- No secret goes in the peer's environment: Kit prints its whole environment into the console
  at every boot, and the console is a log file. The Cesium ion token rides the setup message.
- The peer answers ready only after every camera product has rendered once. On a machine with a
  cold shader cache the RTX renderer comes up minutes after Kit's boot, and a ready before that
  starves the loop. The console prints the wait every 30 s.
- The peer's console is `~/.cache/nexus/logs/console-*.log`: its own lines start `[kit]`, and
  every 240 frames one of them splits a frame's time into the wait for the host, the render and
  the send.
