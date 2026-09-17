"""nexus examples: runnable demos that exercise the framework, controllers and design-opt.

Run one straight from the installed package, with no checkout needed, mirroring NVIDIA Newton's
``python -m newton.examples`` launcher::

    uv run -m nexus.examples --list
    uv run -m nexus.examples sampling_mpc

Every example runs with zero required args by design: its configuration, the vehicle, scene, and
waypoints, lives in the script, and each records its flight ``.rrd``. Some need an optional extra,
for example ``--extra acados`` for ``acados_nmpc`` and ``--extra policy`` for ``goto_policy``, or
take an optional flag, such as ``goto_policy --policy`` and ``px4_sitl --timeout``; the shared
diagnostics flags ``--profile``, ``--trace``, and ``--benchmark`` work on every example. See each
example's module docstring. They all also run RTX-rendered under the Isaac runtime:
``nexus script nexus.examples <name>``.
"""

from __future__ import annotations

# Short name → dotted module path for the `python -m nexus.examples` launcher. Curated rather
# than auto-discovered: it gives stable, readable names, and it disambiguates the two `flight.py`s.
_EXAMPLES = {
    "pid": "nexus.examples.controllers.pid.flight",
    "sampling_mpc": "nexus.examples.controllers.sampling_mpc.obstacle_slalom",
    "acados_nmpc": "nexus.examples.controllers.acados_nmpc.waypoint_tracking",
    "px4_sitl": "nexus.examples.controllers.px4.flight",
    "px4_sitl_manual": "nexus.examples.controllers.px4.flight_manual",
    "px4_mission": "nexus.examples.controllers.px4.mission",
    "goto_policy": "nexus.examples.controllers.policy.goto.flight",
    "gain_tuning": "nexus.examples.design_opt.gain_tuning",
    "mass_recovery": "nexus.examples.design_opt.mass_recovery",
}


def get_examples() -> dict[str, str]:
    """Map each example's short name to its dotted module path."""
    return dict(_EXAMPLES)


def _print_examples() -> None:
    print("Available examples:")
    for name in _EXAMPLES:
        print(f"  {name}")


def main() -> None:
    """Entry point for ``python -m nexus.examples <name> [args...]``."""
    import runpy
    import sys

    examples = get_examples()
    argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help", "--list"):
        print("Usage: python -m nexus.examples <name> [options]")
        print("       python -m nexus.examples --list")
        print()
        _print_examples()
        sys.exit(0)

    name = argv[0]
    if name not in examples:
        print(f"Error: unknown example {name!r}\n")
        _print_examples()
        sys.exit(1)

    # The cross-cutting diagnostics flags --profile, --trace, and --benchmark belong to the
    # launcher, not the example. Configuring them here gives every example the flags without
    # carrying its own, spelled exactly as `nexus run` spells them. argv passes through
    # untouched: an example's own parser, such as px4's --timeout and policy's --policy, uses
    # parse_known_args, and the acados bootstrap re-exec needs the flags to survive in argv because
    # it re-configures after the exec.
    from nexus._src.diagnostics import configure_from_argv

    configure_from_argv(argv[1:])

    # The example gets itself as argv[0] with its own args after, then runs as __main__.
    # alter_sys=True makes runpy set argv[0] to the path of the module's file, not the dotted name,
    # which the examples that re-exec themselves via sys.argv, the acados LD_LIBRARY_PATH
    # bootstrap, require.
    target = examples[name]
    sys.argv = [target, *argv[1:]]
    runpy.run_module(target, run_name="__main__", alter_sys=True)
