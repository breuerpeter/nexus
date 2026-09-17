"""Process-wide run diagnostics: the shared ``--profile`` / ``--trace`` / ``--benchmark`` flags.

Cross-cutting instrumentation toggles, not run *configuration*: they never change what the sim
computes, only what gets measured and reported, so they don't belong in ``LaunchConfig``, which the
tested-config receipt reproduces byte-for-byte. They're process-scoped by nature, one loop, one
profiler, and used to ride env vars: ``NEWTON_PROFILE`` / ``NEWTON_TRACE`` / ``NEWTON_BENCHMARK``.
This module keeps that process-global scope but gives it a typed, documented surface the entry
points fill from parsed args, ``Sim.from_args`` and the examples launcher call :meth:`configure`,
and the instrumented seams read: the orchestrator's profiler, the Isaac ``RtxFrame`` benchmark.

Import-light BY DESIGN, stdlib only: the examples launcher and the command-line tool configure it
before any runtime, or Kit, boots.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Diagnostics:
    """The per-process diagnostics switches; see the module docstring."""

    profile: bool = False
    """Deep loop profiling: periodic per-tick breakdown reports every 5 s, plus carb zone mirroring in Kit."""
    trace: str | None = None
    """Write a Chrome/Perfetto trace of the loop spans to this path on exit."""
    benchmark: bool = False
    """Isaac Sim runtime: attach Isaac's ``benchmark.services`` recorders, ``KitBenchmark``."""

    def configure(self, args: object) -> None:
        """Load the shared flags from a parsed-args namespace, skipping missing attributes.

        OR-merges booleans, so configuring from two entry points, the examples launcher and then
        ``Sim.from_args``, never switches a diagnostic back off.
        """
        self.profile = self.profile or bool(getattr(args, "profile", False))
        self.trace = getattr(args, "trace", None) or self.trace
        self.benchmark = self.benchmark or bool(getattr(args, "benchmark", False))


diagnostics = Diagnostics()
"""The one per-process instance: entry points ``configure()`` it, instrumented seams read it."""


def add_diagnostics_args(p):
    """Register the shared flags on an ``argparse`` parser. Used by ``sim_argparser``, the
    ``nexus`` command-line tool, and the examples launcher, so every entry point spells them
    identically.
    """
    p.add_argument(
        "--profile", action="store_true",
        help="deep loop profiling: periodic per-tick breakdown reports (5 s) + Kit zone mirroring",
    )  # fmt: skip
    p.add_argument(
        "--trace", default=None, metavar="PATH",
        help="write a Chrome/Perfetto trace of the loop spans to PATH on exit",
    )  # fmt: skip
    p.add_argument(
        "--benchmark", action="store_true",
        help="isaacsim runtime: attach Isaac's benchmark.services recorders (KitBenchmark)",
    )  # fmt: skip
    return p


def configure_from_argv(argv: list[str] | None = None) -> None:
    """Configure the process diagnostics from raw ``argv``, default ``sys.argv[1:]``, ignoring
    everything else. For entry points without a full parser: the examples launcher, which passes
    argv through untouched so an example's own parser still receives it, and a script that re-execs
    itself, such as the acados ``LD_LIBRARY_PATH`` bootstrap, where exec replaces the process and
    drops the launcher's configuration, so the flags ride ``argv`` across.
    """
    import argparse
    import sys

    ns, _rest = add_diagnostics_args(argparse.ArgumentParser(add_help=False)).parse_known_args(
        sys.argv[1:] if argv is None else argv
    )
    diagnostics.configure(ns)
