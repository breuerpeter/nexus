"""Run one Kit-only script under a booted Kit app: what ``nexus script`` starts in the Kit image.

The asset scripts in ``scripts/assets/`` that need Kit call its extensions but never boot it, so
this boots Kit headless, hands the target its own arguments, and runs it as ``__main__``. A failure
prints its traceback before Kit's teardown, which can end the process outright and eat both the
traceback and the exit status.
"""

from __future__ import annotations

import os
import runpy
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: script.py <path> [args…]", file=sys.stderr)
        return 2
    argv = sys.argv[1:]
    # SimulationApp reads sys.argv as Kit's own arguments: a target's --out would become one.
    sys.argv = sys.argv[:1]
    from isaacsim import SimulationApp

    app = SimulationApp(
        {"headless": True, "renderer": "RayTracedLighting", "width": 1280, "height": 720, "multi_gpu": False},
        experience="/isaac-sim/apps/isaacsim.exp.full.kit",
    )
    sys.argv = argv  # the target finds itself as argv[0], its own args after
    # The host mounts the working folder at its own path and makes it the container's working folder, so a
    # script run from a checkout imports its siblings the way it did there: scripts.assets.scene_root.
    sys.path.insert(0, os.getcwd())
    code = 0
    try:
        runpy.run_path(argv[0], run_name="__main__")
    except SystemExit as e:  # the script's own sys.exit(n)
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        app.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
