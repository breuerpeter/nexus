"""`nexus run <vehicle>`: thin command-line entry into a runtime.

The wiring + loop ownership live in the runtime packages, architecture.md §12; this is just the
command surface: the PX4 front door, ``run``, plus ``script``, which runs any Python, for example the
zero-arg examples, under the Isaac Sim runtime. ``run`` carries the ``sim_argparser`` flags plus the
runtime-only ``--runtime`` / ``--stream``. ``--runtime``
selects ``standalone``, the lean, headless default that builds a :class:`~nexus._src.api.sim.Sim` and
drives it, or ``isaacsim``, the in-process Isaac Sim runtime with Newton physics plus the RTX and
First Person View (FPV) path. The ``isaacsim`` import is lazy so the standalone path never needs the Kit
env; ``isaacsim`` only exists inside the nvcr.io/nvidia/isaac-sim container.
"""

from __future__ import annotations

import sys

from nexus._src.api.args import sim_argparser  # import-light: parses BEFORE any runtime or Kit boots


def _run_standalone(args) -> None:
    """The standalone launch: build a ``Sim`` from the shared args and drive it to completion. A PX4
    run feeds the peer until it disconnects, the run hits ``max_steps``, or Ctrl-C.
    """
    from nexus._src.api.sim import Sim

    with Sim.from_args(args) as sim:
        sim.run()  # drives setup, which binds :4560 and starts PX4, and then every tick, on this thread


def _run_script(argv: list[str]) -> None:
    """``nexus script <module-or-path> [args…]``: run any Python under the Isaac Sim runtime.

    The script-shaped twin of ``run``'s runtime abstraction: a ``na.Sim`` script needs a booted Kit
    app for its vehicle's RTX sensors to render, and the user shouldn't have to know that. On the
    host this re-launches the *same* command inside the Kit container, as ``--runtime auto`` does
    for ``run``; inside the container it boots ``SimulationApp`` and hands off via ``runpy``.
    ``Sim`` detects the booted Kit and builds through the isaacsim glue, so the same script
    runs rendered here and renderless under plain ``python``. The target receives everything after
    it verbatim; its own argparse owns those args.

        nexus script nexus.examples acados_nmpc --vehicle astro_max_fpv --log
        nexus script my_flight.py --whatever
    """
    if not argv:
        raise SystemExit("usage: nexus script <module-or-path> [args…]")
    try:
        import isaacsim  # noqa: F401  probe: is this the Kit env?
    except ImportError:
        from nexus._src.cli.detect import launch_isaacsim_container

        raise SystemExit(launch_isaacsim_container(["script", *argv], pin_runtime=False)) from None

    import runpy

    from nexus._src.runtimes.isaacsim.runtime import (
        NEWTON_EXPERIENCE,
        apply_pins_post_boot,
        apply_pins_pre_boot,
    )

    # SimulationApp re-reads sys.argv as Kit's args, so the boot must not see the target's flags: a
    # --out path becomes an unknown Kit arg. apply_pins_pre_boot appends the extension exclusion Kit
    # must see, so trim first, then restore argv for the target below.
    sys.argv = sys.argv[:1]
    apply_pins_pre_boot()  # workspace newton/warp pins, a no-op without NEXUS_PINS_DIR
    from isaacsim import SimulationApp

    # Render-capable boot, matching run_isaacsim: multi_gpu off, because the Gaussian-splat NuRec renderer
    # refuses to activate under multi-GPU tiling.
    app = SimulationApp(
        {"headless": True, "renderer": "RayTracedLighting", "width": 1280, "height": 720, "multi_gpu": False},
        experience=NEWTON_EXPERIENCE,
    )
    apply_pins_post_boot()
    sys.argv = argv  # the target finds itself as argv[0], its args after
    code = 0
    try:
        target = argv[0]
        if target.endswith(".py") or "/" in target:
            runpy.run_path(target, run_name="__main__")
        else:
            runpy.run_module(target, run_name="__main__", alter_sys=True)
    except SystemExit as e:  # the script's own sys.exit(n)
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        # Report BEFORE app.close(): Kit's teardown can end the process outright, eating
        # both the traceback and the exit status, so a failed script then looks like a clean rc=0
        # run. That's how the benchmark-matrix cells failed silently in CI, GH #39.
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        code = 1
    finally:
        app.close()
    raise SystemExit(code)


def main() -> None:
    # `script` owns its whole tail, because the target's argparse owns the args, so route it before the sim
    # parser can claim flags such as --vehicle for itself.
    if len(sys.argv) > 1 and sys.argv[1] == "script":
        _run_script(sys.argv[2:])
        return

    parser = sim_argparser(description="Newton framework CLI: run a scenario in a runtime (decoupled TCP:4560 sim).")
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run"],
        help="subcommand (also: `nexus script <module-or-path> [args…]`, run any Python "
        "under the Isaac Sim runtime, auto-launching the Kit container from the host)",
    )
    parser.add_argument(
        "--runtime",
        default="auto",
        choices=["auto", "standalone", "isaacsim"],
        help="runtime. auto (default) reads the vehicle USD: RTX sensor prims (camera/lidar) -> isaacsim "
        "(auto-launching the Kit container from the host), none -> the lean standalone runtime. "
        "There is no --render flag: a camera authored in the USD IS the render decision.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="isaacsim: route the first RTX camera's feed to NVENC → RTSP → MediaMTX → WHEP "
        "(architecture.md §11); without it, frames go to the Rerun recording only.",
    )
    args = parser.parse_args()

    if args.log and args.view:  # serve or file, never both: a live server and a complete .rrd can't coexist in-process
        parser.error("--log (write .rrd) and --view (serve viewer) are mutually exclusive, pick one")

    from nexus._src.cli.detect import launch_isaacsim_container, resolve_runtime

    # 'auto' -> the vehicle Universal Scene Description (USD) decides: RTX sensors mean isaacsim
    runtime = resolve_runtime(args)
    if runtime == "standalone":
        if args.stream:
            parser.error("--stream needs the isaacsim runtime (RTX camera sensors); this vehicle has none")
        _run_standalone(args)
        return
    # isaacsim: inside the Kit env this import works, because the container entrypoint runs this tool under
    # Kit's Python with --runtime isaacsim pinned; on the host it doesn't, so auto-launch the container instead.
    try:
        import isaacsim  # noqa: F401  probe: is this the Kit env?

        from nexus._src.runtimes.isaacsim import run_isaacsim
    except ImportError:
        raise SystemExit(launch_isaacsim_container(sys.argv[1:])) from None

    run_isaacsim(
        vehicle=args.vehicle,
        registry=args.registry,
        control=args.control,
        view=args.view,
        log=args.log,
        debug=args.debug,
        stream=args.stream,
        scene=args.scene,
        geo=args.geo,
        max_steps=args.max_steps,
        rtf=args.rtf,
    )


if __name__ == "__main__":
    main()
