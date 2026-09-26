"""Newton: Freefly's drone simulation framework on NVIDIA Newton physics."""

from typing import TYPE_CHECKING

# Lazy re-exports, per Python Enhancement Proposal (PEP) 562. `import nexus` stays import-light:
# the physics stack imports `newton` and `warp` at module load, which takes seconds, and the
# command-line tool parses its arguments before the run needs either. Each public name resolves on
# first attribute access instead, so `from nexus import Sim` and `nexus.Sim` stay the same
# for callers, and that holds for every name.
_LAZY_EXPORTS = {
    "Sim": "nexus._src.api",
    "sim_argparser": "nexus._src.api",
    "save_run_artifacts": "nexus._src.api",
    "BodyState": "nexus._src.api",
    "JointState": "nexus._src.api",
    "SensorSample": "nexus._src.api",
    "LaunchConfig": "nexus._src.config",
    "Registry": "nexus._src.config",
    "Controls": "nexus._src.core",
    "EnvSample": "nexus._src.core",
    "Measurement": "nexus._src.core",
    "Orchestrator": "nexus._src.core",
    "SimTime": "nexus._src.core",
    # The framework logger. Examples report results through it, with no bare prints: always the
    # console on stderr, plus the recording's logs/sim panel when a Logger is active, through the
    # teed handler.
    "logger": "nexus._src.core",
}

# Literal mirror of _LAZY_EXPORTS keys, kept sorted: ruff PLE0605 wants a list/tuple literal.
__all__ = [
    "BodyState",
    "Controls",
    "EnvSample",
    "JointState",
    "LaunchConfig",
    "Measurement",
    "Orchestrator",
    "Registry",
    "SensorSample",
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
    from nexus._src.api import (
        BodyState,
        JointState,
        SensorSample,
        Sim,
        save_run_artifacts,
        sim_argparser,
    )
    from nexus._src.config import LaunchConfig, Registry
    from nexus._src.core import (
        Controls,
        EnvSample,
        Measurement,
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
