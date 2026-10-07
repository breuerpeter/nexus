"""The program-driving API: ``Sim``, the one public entry to a run.

This is the first slice: a ``Sim`` handle that owns/runs/observes the sim from
Python, with ground truth read off the ``Recorder`` in ``_src/recording``;
``BodyState`` / ``JointState`` / ``SensorSample`` are its public per-entity read types.

Lazy re-exports, per Python Enhancement Proposal (PEP) 562 and mirroring the top-level package: importing
this package, or a light submodule such as ``.args``, must not pull the physics stack: the command-line
tool imports ``api.args`` to parse its arguments before the run builds.
"""

from typing import TYPE_CHECKING

_LAZY = {
    "Sim": "nexus_sim._src.api.sim",
    "sim_argparser": "nexus_sim._src.api.args",
    "save_run_artifacts": "nexus_sim._src.api.args",
    "BodyState": "nexus_sim._src.recording",
    "JointState": "nexus_sim._src.recording",
    "SensorSample": "nexus_sim._src.recording",
}

__all__ = ["BodyState", "JointState", "SensorSample", "Sim", "save_run_artifacts", "sim_argparser"]

# Static mirror of _LAZY for griffe/mkdocstrings; see the top-level nexus_sim/__init__.py. Parsed,
# never executed at runtime, because TYPE_CHECKING is False, so the import-light PEP 562 behavior stays.
if TYPE_CHECKING:
    from nexus_sim._src.api.args import save_run_artifacts, sim_argparser
    from nexus_sim._src.api.sim import Sim
    from nexus_sim._src.recording import BodyState, JointState, SensorSample


def __getattr__(name: str):
    try:
        module = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    return getattr(importlib.import_module(module), name)
