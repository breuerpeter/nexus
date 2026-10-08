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
    GeodeticOrigin,
    LaunchConfig,
    Output,
    Px4Spec,
    Runtime,
)
from .receipt import ResolvedLaunch, TestedConfig
from .resolve import resolve

__all__ = [
    "AssetRef",
    "Catalog",
    "CatalogError",
    "GeodeticOrigin",
    "LaunchConfig",
    "NoMatchError",
    "Output",
    "Px4Spec",
    "ResolvedLaunch",
    "Runtime",
    "Scene",
    "TestedConfig",
    "VehicleVariant",
    "load_catalog",
    "resolve",
]
