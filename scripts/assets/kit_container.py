"""Run a Kit-only asset script in the Kit image: the host side that starts the container, and the
container side that ends the script cleanly.

The three Kit-only scripts beside this module, ``obj_to_usd.py``, ``site_scan_splat.py`` and
``author_cesium_scene.py``, call Kit's extensions, which exist only in the Kit image. Each runs on
the host with plain Python, ``uv run python scripts/assets/<script>.py …``, and on the host
:func:`run_in_kit` starts the image with the same file and arguments, where the script boots Kit
itself and does its work. :func:`in_kit` says which side this process runs on.

The container mounts the working folder read-write at its host path and works in it, and mounts
``$NEXUS_DATA``, default ``~/data``, where scans and converted scenes live, the same way: a path
typed on the host means the same file inside. The console streams to this terminal, and the
script's exit code is the command's.

The image is the render peer's, NVIDIA's Isaac Sim as pulled, so this module takes its pull, its
run options, its Cesium mount and its docker client from ``nexus._src.rendering.peer``, private
names that a peer change can move.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

IN_KIT = "NEXUS_IN_KIT"  # set in the container's environment, so a script knows which side it runs on

# The Kit app a script boots: headless, with the renderer the converters' extensions expect.
KIT_APP = {"headless": True, "renderer": "RayTracedLighting", "width": 1280, "height": 720, "multi_gpu": False}
KIT_EXPERIENCE = "/isaac-sim/apps/isaacsim.exp.full.kit"


def in_kit() -> bool:
    """Whether this process runs inside the Kit container started by :func:`run_in_kit`."""
    return os.environ.get(IN_KIT) == "1"


def kit_argv() -> list[str]:
    """The script's own arguments, taken off ``sys.argv`` before the boot: Kit reads ``sys.argv`` as its
    own arguments, so a script's ``--out`` would become one.
    """
    argv = sys.argv[1:]
    sys.argv = sys.argv[:1]
    return argv


def run_in_kit(script: str | os.PathLike, args: list[str], *, cesium: bool = False) -> int:
    """Run ``script`` with ``args`` in NVIDIA's Isaac Sim image, and return its exit code.

    A machine pulls the image once. With ``cesium``, the host fetches the Cesium for Omniverse
    extensions into the asset cache first and mounts them, for the script that authors the Cesium scene.
    A failure to reach Docker, pull the image, fetch Cesium or start the container prints one line
    that names the cause and returns 1: no traceback, since nothing of the script ran.
    """
    from docker.errors import DockerException

    import nexus._src.rendering.peer as peer
    from nexus._src.assets.resolver import default_cache

    target = Path(script).resolve()
    cwd = Path.cwd()
    try:
        volumes, env = peer._cesium_mount(default_cache()) if cesium else ({}, {})
        image = peer.ensure_image()
        volumes[str(cwd)] = {"bind": str(cwd), "mode": "rw"}
        if not target.is_relative_to(cwd):
            volumes[str(target.parent)] = {"bind": str(target.parent), "mode": "ro"}
        data = Path(os.environ.get("NEXUS_DATA") or Path.home() / "data").resolve()
        if data.is_dir() and not data.is_relative_to(cwd):
            volumes[str(data)] = {"bind": str(data), "mode": "rw"}
        env[IN_KIT] = "1"
        if os.environ.get("NEXUS_DATA"):
            env["NEXUS_DATA"] = os.environ["NEXUS_DATA"]
        container = peer.client().containers.run(
            image,
            command=[str(target), *args],  # the run options make Kit's python the entrypoint
            working_dir=str(cwd),
            labels={peer.LABEL: "script"},
            detach=True,
            **peer._run_options(volumes, env),
        )
    except DockerException as exc:
        print(
            f"{target.name}: starting the Kit container failed: {getattr(exc, 'explanation', None) or exc}",
            file=sys.stderr,
        )
        return 1
    except RuntimeError as exc:  # KitPeerError included: the pull, the Cesium fetch, or the daemon is unreachable
        print(f"{target.name}: the script runs in a Kit container, but {exc}", file=sys.stderr)
        return 1
    try:
        for chunk in container.logs(stream=True, follow=True):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        return int(container.wait().get("StatusCode", 1))
    finally:
        container.remove(force=True)  # Ctrl-C included: the container dies with this command


def finish(app, main, argv: list[str]) -> None:
    """Run ``main(argv)`` under the booted Kit ``app`` and exit with its code.

    A failure prints its traceback and flushes, then ends the process with its code at once: Kit's
    teardown ends the process with status 0 whatever the script's code was, so a failure never
    reaches ``app.close()``. A success closes Kit cleanly. ``main`` ends by returning an int, by
    returning None for 0, or by raising ``SystemExit``.
    """
    code = 0
    try:
        out = main(argv)
        code = out if isinstance(out, int) else 0
    except SystemExit as e:  # the script's own sys.exit(n)
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        import traceback

        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    if code:
        os._exit(code)
    app.close()
    sys.exit(0)
