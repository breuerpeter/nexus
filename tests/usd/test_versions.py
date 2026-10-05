"""A schema changes only by a new version, and the build fails a version no plugin defines.

The stand-in project under `stand_in/`, which the root conftest registers, defines the family
`StandInGearAPI` at two versions and records `StandInOldAPI` and `StandInOldAPI_1` as retired.
"""

import shutil
from importlib.metadata import version
from pathlib import Path

import pytest

from nexus._src.build.components import resolve_components
from nexus._src.core.registry import ComponentRegistry

FIXTURE = Path(__file__).with_name("conformance.usda")
PLUGIN = Path(__file__).resolve().parents[2] / "nexus" / "_src" / "usd"
IMU = "/Vehicle/body/Imu"
ACC_NOISE = "float nexus:accNoise = 0.02"


class First:
    """A stand-in class for the first version of a family."""


class Second:
    """A stand-in class for the second version of a family."""


def _vehicle(tmp_path, body: str) -> Path:
    """A vehicle file whose root prim holds `body`."""
    vehicle = tmp_path / "vehicle.usda"
    vehicle.write_text(f'#usda 1.0\n(\n    defaultPrim = "Vehicle"\n)\n\ndef Xform "Vehicle"\n{{\n{body}\n}}\n')
    return vehicle


def _fixture_with(tmp_path, old: str, new: str) -> Path:
    """A copy of the conformance fixture with `old` replaced by `new`."""
    text = FIXTURE.read_text()
    assert old in text
    vehicle = tmp_path / "vehicle.usda"
    vehicle.write_text(text.replace(old, new))
    return vehicle


def _plugin_with(tmp_path, old: str, new: str) -> Path:
    """A copy of the nexus plugin folder with `old` replaced by `new` in its schema file."""
    copy = tmp_path / "plugin"
    copy.mkdir()
    shutil.copy(PLUGIN / "plugInfo.json", copy)
    text = (PLUGIN / "generatedSchema.usda").read_text()
    assert old in text
    (copy / "generatedSchema.usda").write_text(text.replace(old, new))
    return copy


def test_two_versions_of_one_family_build_side_by_side_each_its_own_class(tmp_path):
    """Two versions of one schema family build side by side, each its own class with its own version's keyword arguments.

    Given a stand-in plugin that defines a family at two versions with a renamed attribute between them,
    and a registry that maps each version to a stand-in class, when a fixture applies one version on each
    of two prims and the build resolves it, then each prim resolves to its version's class with that
    version's keywords.
    """
    vehicle = _vehicle(
        tmp_path,
        '    def Xform "a" (prepend apiSchemas = ["StandInGearAPI"])\n    {\n        float nexus:ratio = 2\n    }\n'
        '    def Xform "b" (prepend apiSchemas = ["StandInGearAPI_1"])\n    {\n        float nexus:gearRatio = 3\n    }',
    )
    specs = resolve_components(vehicle, ComponentRegistry({"StandInGearAPI": First, "StandInGearAPI_1": Second}))
    assert [(spec.prim, spec.cls, spec.kwargs) for spec in specs] == [
        ("/Vehicle/a", First, {"ratio": 2.0}),
        ("/Vehicle/b", Second, {"gear_ratio": 3.0}),
    ]


def test_an_undefined_version_of_a_defined_family_fails_the_build(tmp_path):
    """A prim that applies an undefined version of a family a plugin defines fails the build, naming the prim, the version it applies and the framework version.

    Given the fixture's root prim applying `NexusPx4SitlAPI_9` and authoring no `nexus:` attribute, when
    resolved, then the build fails and the message carries the prim path, `NexusPx4SitlAPI_9` and the
    version of the installed framework.
    """
    vehicle = _fixture_with(tmp_path, '"NexusPx4SitlAPI"', '"NexusPx4SitlAPI_9"')
    with pytest.raises(ValueError) as e:
        resolve_components(vehicle, ComponentRegistry({"NexusImuAPI": First, "NexusPx4API": First}))
    assert all(text in str(e.value) for text in ("/Vehicle", "NexusPx4SitlAPI_9", version("nexus-sim")))


def test_an_undefined_version_with_its_attributes_authored_fails_naming_the_version(tmp_path):
    """The same failure names the version when the prim also authors the schema's attributes.

    Given a prim applying `NexusImuAPI_9` with `nexus:accNoise` authored, when resolved, then the build
    fails naming the prim, `NexusImuAPI_9` and the framework version, not an undefined attribute.
    """
    vehicle = _fixture_with(tmp_path, '"NexusImuAPI"', '"NexusImuAPI_9"')
    with pytest.raises(ValueError) as e:
        resolve_components(vehicle, ComponentRegistry({"NexusPx4API": First}))
    assert all(text in str(e.value) for text in (IMU, "NexusImuAPI_9", version("nexus-sim")))


def test_a_version_of_a_fully_retired_family_fails_the_build(tmp_path):
    """A version of a family with every version retired fails the same way.

    Given a stand-in plugin that has retired every version of a family, and a fixture prim applying one of
    them, when resolved, then the build fails naming the prim, the version and the framework version.
    """
    vehicle = _vehicle(tmp_path, '    def Xform "old" (prepend apiSchemas = ["StandInOldAPI_1"])\n    {\n    }')
    with pytest.raises(ValueError) as e:
        resolve_components(vehicle, ComponentRegistry({}))
    assert all(text in str(e.value) for text in ("/Vehicle/old", "StandInOldAPI_1", version("nexus-sim")))


def test_a_schema_of_a_family_no_plugin_ever_defined_is_skipped(tmp_path):
    """A schema of a family no registered plugin has ever defined is still skipped.

    Given a fixture prim applying `AcmeFooAPI`, which no plugin defines or has retired, when resolved, then
    the build returns no component for it and raises nothing.
    """
    vehicle = _vehicle(tmp_path, '    def Xform "acme" (prepend apiSchemas = ["AcmeFooAPI"])\n    {\n    }')
    assert resolve_components(vehicle, ComponentRegistry({})) == []


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (ACC_NOISE, "float nexus:accelNoise = 0.02"),
        (ACC_NOISE, "double nexus:accNoise = 0.02"),
        ("Units: meters per second squared", "Units: standard gravities"),
    ],
    ids=["renamed", "retyped", "another unit"],
)
def test_a_released_attribute_changed_in_place_fails_the_check(tmp_path, old, new):
    """A released version that loses or renames an attribute, or changes one's type or unit, fails the suite.

    Given the committed record of the released versions and a copy of the plugin with one released
    attribute renamed, then retyped, then given another unit, when the check runs on each copy, then each
    fails naming the schema version and the attribute.
    """
    from nexus._src.usd.released import in_place_changes

    changes = in_place_changes(_plugin_with(tmp_path, old, new))
    assert [change for change in changes if "NexusImuAPI" in change and "nexus:accNoise" in change]


def test_an_attribute_that_joins_a_released_version_with_a_fallback_passes_the_check(tmp_path):
    """An attribute that joins a released version with a fallback passes.

    Given the committed record and a copy of the plugin with one new attribute with a fallback on a
    released version, when the check runs, then it passes.
    """
    from nexus._src.usd.released import in_place_changes

    joined = f'float nexus:accBias = 0 (\n        doc = "A new attribute."\n    )\n    {ACC_NOISE}'
    assert in_place_changes(_plugin_with(tmp_path, ACC_NOISE, joined)) == []
