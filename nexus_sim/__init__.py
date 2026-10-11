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
    # The run: its one entry, the argument parser the scripts share and the artifacts helper.
    "Sim": "nexus_sim._src.api",
    "sim_argparser": "nexus_sim._src.api",
    "save_run_artifacts": "nexus_sim._src.api",
    # Configuration: the vehicle and scene catalog, and the launch config.
    "Catalog": "nexus_sim._src.config",
    "LaunchConfig": "nexus_sim._src.config",
    # The component contract and the types its stages speak.
    "Component": "nexus_sim._src.core.interfaces",
    "Stage": "nexus_sim._src.core.interfaces",
    "Tick": "nexus_sim._src.core.interfaces",
    "Signal": "nexus_sim._src.core.signals",
    # The registry, a value a run takes, and its registration call.
    "Registry": "nexus_sim._src.core.registry",
    "register": "nexus_sim._src.core.registry",
    # The peer contract.
    "Peer": "nexus_sim._src.peers",
    # The signal value types a component reads or writes: the tick's time, the estimate, the setpoints
    # and each shipped sensor's sample.
    "SimTime": "nexus_sim._src.core.schema",
    "PoseTwist": "nexus_sim._src.core.schema",
    "PositionGoal": "nexus_sim._src.core.schema",
    "Waypoints": "nexus_sim._src.core.schema",
    "ReferenceTrajectory": "nexus_sim._src.core.schema",
    "ImuSample": "nexus_sim._src.core.schema",
    "MagSample": "nexus_sim._src.core.schema",
    "BaroSample": "nexus_sim._src.core.schema",
    "GpsSample": "nexus_sim._src.core.schema",
    "Image": "nexus_sim._src.core.schema",
    "PointCloud": "nexus_sim._src.core.schema",
    # The framework logger. Examples report results through it, with no bare prints: always the
    # console on stderr, plus the recording's Logs pane, at sim/logs/<module>, when a Logger is active,
    # through the teed handler.
    "logger": "nexus_sim._src.core",
}

# Literal mirror of _LAZY_EXPORTS keys, kept sorted: ruff PLE0605 wants a list/tuple literal.
__all__ = [
    "BaroSample",
    "Catalog",
    "Component",
    "GpsSample",
    "Image",
    "ImuSample",
    "LaunchConfig",
    "MagSample",
    "Peer",
    "PointCloud",
    "PoseTwist",
    "PositionGoal",
    "ReferenceTrajectory",
    "Registry",
    "Signal",
    "Sim",
    "SimTime",
    "Stage",
    "Tick",
    "Waypoints",
    "logger",
    "register",
    "save_run_artifacts",
    "sim_argparser",
]

# Static mirror of _LAZY_EXPORTS for type checkers and griffe/mkdocstrings: the API-reference
# build resolves `::: nexus_sim.<Name>` through these aliases. Parsed but never executed at
# runtime, because TYPE_CHECKING is False, so the preceding import-light PEP 562 behavior stays.
if TYPE_CHECKING:
    from nexus_sim._src.api import (
        Sim,
        save_run_artifacts,
        sim_argparser,
    )
    from nexus_sim._src.config import Catalog, LaunchConfig
    from nexus_sim._src.core import logger
    from nexus_sim._src.core.interfaces import Component, Stage, Tick
    from nexus_sim._src.core.registry import Registry, register
    from nexus_sim._src.core.schema import (
        BaroSample,
        GpsSample,
        Image,
        ImuSample,
        MagSample,
        PointCloud,
        PoseTwist,
        PositionGoal,
        ReferenceTrajectory,
        SimTime,
        Waypoints,
    )
    from nexus_sim._src.core.signals import Signal
    from nexus_sim._src.peers import Peer


def __getattr__(name: str):
    try:
        module = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    return getattr(importlib.import_module(module), name)


def __dir__():
    return __all__
