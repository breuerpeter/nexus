"""`nexus run <vehicle>`: the thin command-line entry into a run.

``run`` builds a :class:`~nexus._src.api.sim.Sim` from the shared ``sim_argparser`` flags plus
``--stream`` and drives it to completion. A vehicle whose Universal Scene Description (USD) file
authors RTX sensor prims renders them in the Kit render peer, a container the run starts from this
host.
"""

from __future__ import annotations

from nexus._src.api.args import sim_argparser  # import-light: parses before the run builds


def _run(args) -> None:
    """Build a ``Sim`` from the shared args and drive it to completion. A PX4 run feeds the peer
    until it disconnects, the run hits ``max_steps``, or Ctrl-C.
    """
    from nexus._src.api.sim import Sim

    with Sim.from_args(args) as sim:
        sim.run()  # drives setup, which binds :4560 and starts PX4, and then every tick, on this thread


def main() -> None:
    parser = sim_argparser(description="Fly a vehicle against PX4 SITL over the HIL link, TCP:4560.")
    parser.add_argument("command", nargs="?", default="run", choices=["run"], help="subcommand")
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
