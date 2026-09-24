"""`nexus run <vehicle>`: the thin command-line entry into a run.

``run`` builds a :class:`~nexus._src.api.sim.Sim` from the shared ``sim_argparser`` flags plus
``--stream`` and drives it to completion. A vehicle whose Universal Scene Description (USD) file
authors RTX sensor prims renders them in the Kit render peer, a container the run starts from this
host. ``script`` runs one Kit-only asset script in the Kit image.
"""

from __future__ import annotations

import sys

from nexus._src.api.args import sim_argparser  # import-light: parses before the run builds


def _run(args) -> None:
    """Build a ``Sim`` from the shared args and drive it to completion. A PX4 run feeds the peer
    until it disconnects, the run hits ``max_steps``, or Ctrl-C.
    """
    from nexus._src.api.sim import Sim

    with Sim.from_args(args) as sim:
        sim.run()  # drives setup, which binds :4560 and starts PX4, and then every tick, on this thread


def _run_script(argv: list[str]) -> None:
    """``nexus script <path> [args…]``: run one Kit-only script in the Kit image.

    For the asset scripts that call Kit's extensions: the image's launcher boots Kit and runs the
    script with its own arguments, with the working folder and ``$NEXUS_DATA`` mounted at their
    host paths::

        nexus script scripts/assets/obj_to_usd.py "$NEXUS_DATA/scans/obj/Example Site" --out out.usdz
    """
    if not argv:
        raise SystemExit("usage: nexus script <path> [args…]")
    from nexus._src.rendering import KitPeerError, run_script

    try:
        code = run_script(argv)
    except KitPeerError as exc:
        raise SystemExit(f"nexus: {exc}") from None
    raise SystemExit(code)


def main() -> None:
    # `script` owns its whole tail, because the target's argparse owns the args, so route it before the sim
    # parser can claim flags such as --vehicle for itself.
    if len(sys.argv) > 1 and sys.argv[1] == "script":
        _run_script(sys.argv[2:])
        return

    parser = sim_argparser(description="Fly a vehicle against PX4 SITL over the HIL link, TCP:4560.")
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run"],
        help="subcommand (also: `nexus script <path> [args…]`, run a Kit-only asset script in the Kit image)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="publish each RTX camera's feed: NVENC → RTSP → MediaMTX → WHEP (architecture.md §11); "
        "without it, frames go to the Rerun recording only",
    )
    args = parser.parse_args()

    if args.log and args.view:  # serve or file, never both: a live server and a complete .rrd can't coexist in-process
        parser.error("--log (write .rrd) and --view (serve viewer) are mutually exclusive, pick one")

    from nexus._src.rendering import KitPeerError

    try:
        _run(args)
    except KitPeerError as exc:  # the Kit container couldn't start: say why, no traceback
        raise SystemExit(f"nexus: {exc}") from None


if __name__ == "__main__":
    main()
