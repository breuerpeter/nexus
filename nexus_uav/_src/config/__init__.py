"""The framework launch interface.

A static, declarative ``LaunchConfig``, what the sim *is*, resolved by name against a
checked-in, content-addressed ``Catalog`` into a ``TestedConfig`` receipt, what the run
actually simulated. Dynamics such as faults and actors are control-API verbs, not config.
"""

from .catalog import (
    Catalog,
    CatalogError,
    NoMatchError,
    Scene,
    VehicleVariant,
    load_catalog,
)
from .models import (
    AssetRef,
    LaunchConfig,
    Output,
    Runtime,
)
from .receipt import ResolvedLaunch, TestedConfig
from .resolve import resolve

__all__ = [
    "AssetRef",
    "Catalog",
    "CatalogError",
    "LaunchConfig",
    "NoMatchError",
    "Output",
    "ResolvedLaunch",
    "Runtime",
    "Scene",
    "TestedConfig",
    "VehicleVariant",
    "load_catalog",
    "resolve",
]
