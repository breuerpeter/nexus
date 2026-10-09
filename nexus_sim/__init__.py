"""Newton: Freefly's drone simulation framework on NVIDIA Newton physics."""

from typing import TYPE_CHECKING

from nexus_sim._src.usd import register_plugins as _register_plugins

# The one eager step: OpenUSD builds its schema registry once, on first use, and a plugin registered
# after that never shows in it, so the schema plugins register here, before a caller can open a stage.
_register_plugins()

# Lazy re-exports, per Python Enhancement Proposal (PEP) 562. `import nexus_sim` stays import-light:
# the physics stack imports `newton` and `warp` at module load, which takes seconds, and the
# command-line tool parses its arguments before the run needs either. Each public name resolves on
# first attribute access instead, so `from nexus_sim import Sim` and `nexus_sim.Sim` stay the same
# for callers, and that holds for every name.
_LAZY_EXPORTS = {
    "Sim": "nexus_sim._src.api",
    "sim_argparser": "nexus_sim._src.api",
    "save_run_artifacts": "nexus_sim._src.api",
    "BodyState": "nexus_sim._src.api",
    "JointState": "nexus_sim._src.api",
    "Catalog": "nexus_sim._src.config",
    "LaunchConfig": "nexus_sim._src.config",
    "Controls": "nexus_sim._src.core",
    "Orchestrator": "nexus_sim._src.core",
    "SimTime": "nexus_sim._src.core",
    # The framework logger. Examples report results through it, with no bare prints: always the
    # console on stderr, plus the recording's Logs pane, at sim/logs/<module>, when a Logger is active,
    # through the teed handler.
    "logger": "nexus_sim._src.core",
}

# Literal mirror of _LAZY_EXPORTS keys, kept sorted: ruff PLE0605 wants a list/tuple literal.
__all__ = [
    "BodyState",
    "Catalog",
    "Controls",
    "JointState",
    "LaunchConfig",
    "Orchestrator",
    "Sim",
    "SimTime",
    "logger",
    "save_run_artifacts",
    "sim_argparser",
]

# Static mirror of _LAZY_EXPORTS for type checkers and griffe/mkdocstrings: the API-reference
# build resolves `::: nexus.<Name>` through these aliases. Parsed but never executed at
# runtime, because TYPE_CHECKING is False, so the preceding import-light PEP 562 behavior stays.
if TYPE_CHECKING:
    from nexus_sim._src.api import (
        BodyState,
        JointState,
        Sim,
        save_run_artifacts,
        sim_argparser,
    )
    from nexus_sim._src.config import Catalog, LaunchConfig
    from nexus_sim._src.core import (
        Controls,
        Orchestrator,
        SimTime,
        logger,
    )


def __getattr__(name: str):
    try:
        module = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    return getattr(importlib.import_module(module), name)


def __dir__():
    return __all__
