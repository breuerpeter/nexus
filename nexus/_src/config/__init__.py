"""newton-config: the framework launch interface.

A static, declarative ``LaunchConfig``, what the sim *is*, resolved by name against a
checked-in, content-addressed ``Registry`` into a ``TestedConfig`` receipt, what the run
actually simulated. Dynamics such as faults and actors are control-API verbs, not config.
"""

from .models import (
    AssetRef,
    Control,
    Environment,
    GeodeticOrigin,
    LaunchConfig,
    Output,
    Px4Spec,
    Runtime,
)
from .receipt import ResolvedLaunch, TestedConfig
from .registry import (
    Defaults,
    NoMatchError,
    Registry,
    RegistryError,
    Scene,
    VehicleVariant,
    load_registry,
)
from .resolve import resolve

__all__ = [
    "AssetRef",
    "Control",
    "Defaults",
    "Environment",
    "GeodeticOrigin",
    # launch schema
    "LaunchConfig",
    "NoMatchError",
    "Output",
    "Px4Spec",
    # registry
    "Registry",
    "RegistryError",
    "ResolvedLaunch",
    "Runtime",
    "Scene",
    "TestedConfig",
    "VehicleVariant",
    "load_registry",
    # resolution
    "resolve",
]
