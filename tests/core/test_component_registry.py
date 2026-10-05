"""The component registry: a value the builder takes, its default filled from the entry-point group.

A test that adds to the default registry, or installs a package into the group, runs in a child process,
so no test mutates module state. The stand-in project under `tests/usd/stand_in/` supplies the schema,
`StandInAPI`, and the class the child builds; `PYTHONPATH` makes it importable there as `nexus_stand_in`.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAND_IN = ROOT / "tests" / "usd" / "stand_in"
FIXTURE = ROOT / "tests" / "usd" / "conformance.usda"

# Authors two vehicles: one whose one prim applies the stand-in schema, by name, so it needs no plugin,
# and a bare one that applies no schema at all.
_AUTHOR = """
from pxr import Sdf, Usd, UsdGeom

def author_stand_in(path):
    stage = Usd.Stage.CreateNew(str(path))
    prim = UsdGeom.Xform.Define(stage, "/Vehicle/body/Ai").GetPrim()
    prim.AddAppliedSchema("StandInAPI")
    prim.CreateAttribute("nexus:standInGain", Sdf.ValueTypeNames.Float).Set(2.5)
    stage.GetRootLayer().Save()

def author_bare(path):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/Vehicle/body")
    stage.GetRootLayer().Save()
"""


def _python(code: str, *, cwd: Path, pythonpath: Path) -> list[str]:
    """Run `code` in a child process and return the words on the lines it printed after a `>` marker."""
    env = {**os.environ, "PYTHONPATH": str(pythonpath)}
    r = subprocess.run(
        [sys.executable, "-c", _AUTHOR + code], check=False, cwd=cwd, env=env, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    return [word for line in r.stdout.splitlines() if line.startswith("> ") for word in line[2:].split()]


def test_a_registry_resolves_a_schema_to_the_class_it_maps():
    """A registry resolves a schema to the class it maps."""
    from nexus._src.core.registry import ComponentRegistry

    class StandIn:
        """A stand-in component class."""

    assert ComponentRegistry({"StandInAPI": StandIn}).resolve("StandInAPI") is StandIn


def test_a_registry_resolves_an_import_path_to_the_class_it_names():
    """A registry resolves an entry given as an import path, `module:Class`, to the class it names."""
    import json

    from nexus._src.core.registry import ComponentRegistry

    assert ComponentRegistry({"JsonAPI": "json:JSONDecoder"}).resolve("JsonAPI") is json.JSONDecoder


def test_a_registry_resolves_an_entry_added_after_its_creation():
    """A registry resolves an entry that `add` gives it after its creation."""
    from nexus._src.core.registry import ComponentRegistry

    class StandIn:
        """A stand-in component class."""

    registry = ComponentRegistry()
    registry.add("StandInAPI", StandIn)
    assert registry.resolve("StandInAPI") is StandIn


def test_an_installed_package_with_an_entry_in_the_group_builds_its_class_with_no_call(tmp_path):
    """An installed package with an entry in the entry-point group builds its class with no registration call.

    Given a stand-in package under `tests/` installed with an entry in the group, when a run resolves a
    prim applying its schema with the default registry, then the prim resolves to the package's class.
    """
    site = tmp_path / "site"
    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable, "--no-deps", "--target", str(site), str(STAND_IN)],
        check=True,
        capture_output=True,
    )
    vehicle = tmp_path / "vehicle.usda"
    code = f"""
import nexus
from nexus._src.build.components import resolve_components
author_stand_in({str(vehicle)!r})
(ai,) = resolve_components({str(vehicle)!r})
print(">", ai.cls.__module__, ai.cls.__name__)
"""
    assert _python(code, cwd=tmp_path, pythonpath=site) == ["nexus_stand_in.heavy", "StandIn"]


def test_a_project_adds_an_entry_to_the_default_registry_through_a_public_registration_call(tmp_path):
    """A project adds an entry to the default registry through a public registration call.

    Given a process that calls the registration for a schema and a class, when the build resolves a prim
    applying it with the default registry, then the prim resolves to that class.
    """
    vehicle = tmp_path / "vehicle.usda"
    code = f"""
import nexus_stand_in  # registers the stand-in schema plugin
import nexus
from nexus._src.core.registry import register_component
from nexus._src.build.components import resolve_components
from nexus_stand_in.heavy import StandIn
register_component("StandInAPI", StandIn)
author_stand_in({str(vehicle)!r})
(ai,) = resolve_components({str(vehicle)!r})
print(">", ai.cls is StandIn)
"""
    assert _python(code, cwd=tmp_path, pythonpath=STAND_IN) == ["True"]


def test_a_registry_handed_to_the_builder_replaces_the_default_and_leaves_it_untouched():
    """A registry handed to the builder replaces the default and leaves it untouched.

    Given a registry of the test's own mapping the Inertial Measurement Unit (IMU) schema to a stand-in,
    when resolved, then the prim resolves to the stand-in, and a later build with the default resolves it
    to the IMU class.
    """
    from nexus._src.build.components import resolve_components
    from nexus._src.core.registry import ComponentRegistry
    from nexus._src.usd import schema_names
    from nexus._src.vehicle.sensors import ImuSensor

    class StandIn:
        """A stand-in component class."""

    imu_prim = "/Vehicle/body/Imu"
    (stand_in,) = [
        spec
        for spec in resolve_components(FIXTURE, ComponentRegistry(dict.fromkeys(schema_names(), StandIn)))
        if spec.prim == imu_prim
    ]
    (imu,) = [spec for spec in resolve_components(FIXTURE) if spec.prim == imu_prim]
    assert (stand_in.cls, imu.cls) == (StandIn, ImuSensor)


def test_the_build_imports_an_entry_class_only_when_a_run_declares_its_schema(tmp_path):
    """The build imports an entry's class, and only when a run declares its schema.

    Given an entry whose import path is a module that records its import, when a run resolves a vehicle
    that doesn't apply its schema, then the module stays unimported, and when one does, the build imports it.
    """
    bare, vehicle = tmp_path / "bare.usda", tmp_path / "vehicle.usda"
    code = f"""
import sys
import nexus_stand_in  # registers the stand-in schema plugin
import nexus
from nexus._src.core.registry import register_component
from nexus._src.build.components import resolve_components
register_component("StandInAPI", "nexus_stand_in.heavy:StandIn")
author_bare({str(bare)!r})
resolve_components({str(bare)!r})
print(">", "nexus_stand_in.heavy" in sys.modules)
author_stand_in({str(vehicle)!r})
resolve_components({str(vehicle)!r})
print(">", "nexus_stand_in.heavy" in sys.modules)
"""
    assert _python(code, cwd=tmp_path, pythonpath=STAND_IN) == ["False", "True"]


def test_the_provider_registry_goes_and_nexus_registry_still_names_the_catalog():
    """The provider registry nothing calls goes, and `nexus.Registry` still names the catalog.

    Given `import nexus`, when the test looks, then `nexus._src.core` exports no `Registry`
    and `nexus.Registry` resolves the catalog.
    """
    import nexus as na
    import nexus._src.core as core
    from nexus._src.config import Registry as Catalog

    assert (hasattr(core, "Registry"), na.Registry) == (False, Catalog)
