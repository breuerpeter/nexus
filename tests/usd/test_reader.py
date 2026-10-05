"""The build resolves each prim's applied nexus schemas to a class and its keyword arguments, and creates nothing.

The stand-in schema, `StandInAPI`, comes from the project under `stand_in/`, which the root conftest registers.
"""

import shutil
from pathlib import Path

import pytest
from pxr import Sdf, Usd, UsdGeom

from nexus._src.build.components import resolve_components
from nexus._src.core.registry import ComponentRegistry
from nexus._src.usd import schema_names

FIXTURE = Path(__file__).with_name("conformance.usda")
IMU = "/Vehicle/body/Imu"


class StandIn:
    """A stand-in component class."""


def _imu(specs: list) -> list:
    """The specs the fixture's Inertial Measurement Unit (IMU) prim declares."""
    return [spec for spec in specs if spec.prim == IMU]


def _fixture_copy(tmp_path) -> Path:
    copy = tmp_path / "vehicle.usda"
    shutil.copy(FIXTURE, copy)
    return copy


def test_the_reader_names_each_schema_attribute_in_snake_case_without_its_namespace():
    """The reader names each `nexus:` attribute of a prim's schema in snake case, without the namespace."""
    from nexus._src.usd.reader import read_declarations

    kwargs = {prim: kwargs for prim, _, kwargs in read_declarations(FIXTURE)}[IMU]
    assert sorted(kwargs) == ["acc_noise", "gyro_noise", "rate"]


def test_the_reader_fails_on_an_asset_path_that_resolves_to_no_file_and_names_the_prim(tmp_path):
    """The reader fails on an authored asset path that resolves to no file, and names the prim."""
    from nexus._src.usd.reader import read_declarations

    vehicle = tmp_path / "vehicle.usda"
    stage = Usd.Stage.CreateNew(str(vehicle))
    prim = UsdGeom.Xform.Define(stage, "/Vehicle/body/Ai").GetPrim()
    prim.ApplyAPI("StandInAPI")
    prim.GetAttribute("nexus:weights").Set(Sdf.AssetPath("missing.pt"))
    stage.GetRootLayer().Save()

    with pytest.raises(ValueError, match="/Vehicle/body/Ai"):
        read_declarations(vehicle)


def test_the_reader_fails_on_a_nexus_attribute_on_a_prim_with_no_nexus_schema_and_names_the_prim(tmp_path):
    """The reader fails on an authored `nexus:` attribute on a prim that applies no nexus schema."""
    from nexus._src.usd.reader import read_declarations

    vehicle = tmp_path / "vehicle.usda"
    stage = Usd.Stage.CreateNew(str(vehicle))
    prim = UsdGeom.Xform.Define(stage, "/Vehicle/body/Imu").GetPrim()
    prim.CreateAttribute("nexus:accNoise", Sdf.ValueTypeNames.Float).Set(0.05)
    stage.GetRootLayer().Save()

    with pytest.raises(ValueError, match="/Vehicle/body/Imu"):
        read_declarations(vehicle)


def test_each_attribute_the_schema_defines_reaches_the_class_as_the_snake_case_keyword():
    """Each `nexus:` attribute the schema defines reaches the class as the snake-case keyword argument of the same name.

    Given the fixture prim applying the Inertial Measurement Unit (IMU) schema with a two-word attribute
    authored, when the build resolves it with a registry that maps the schema to a stand-in, then the
    stand-in's keyword arguments hold the snake-case name with the authored value.
    """
    (imu,) = _imu(resolve_components(FIXTURE, ComponentRegistry(dict.fromkeys(schema_names(), StandIn))))
    assert imu.kwargs["acc_noise"] == pytest.approx(0.05)


def test_an_attribute_the_prim_does_not_author_takes_the_fallback_the_plugin_defines():
    """An attribute the prim doesn't author takes the fallback the plugin defines.

    Given the fixture prim with one attribute unauthored, when resolved, then the stand-in's keyword
    arguments hold that name at the plugin's fallback.
    """
    (imu,) = _imu(resolve_components(FIXTURE, ComponentRegistry(dict.fromkeys(schema_names(), StandIn))))
    assert imu.kwargs["gyro_noise"] == pytest.approx(0.02)


def test_an_asset_path_attribute_resolves_against_the_layer_that_authored_it(tmp_path):
    """An asset-path attribute resolves against the layer that authored it.

    Given a layer under a subfolder that authors a relative asset path and a vehicle that sublayers it,
    when resolved, then the keyword is the absolute path beside the authoring layer.
    """
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "weights.pt").write_bytes(b"")
    mount = Usd.Stage.CreateNew(str(sub / "mount.usda"))
    prim = UsdGeom.Xform.Define(mount, "/Vehicle/body/Ai").GetPrim()
    prim.ApplyAPI("StandInAPI")
    prim.GetAttribute("nexus:weights").Set(Sdf.AssetPath("weights.pt"))
    mount.GetRootLayer().Save()
    vehicle = tmp_path / "vehicle.usda"
    root = Usd.Stage.CreateNew(str(vehicle))
    root.SetDefaultPrim(UsdGeom.Xform.Define(root, "/Vehicle").GetPrim())
    root.GetRootLayer().subLayerPaths.append("sub/mount.usda")
    root.GetRootLayer().Save()

    (ai,) = resolve_components(vehicle, ComponentRegistry({"StandInAPI": StandIn}))
    assert Path(ai.kwargs["weights"]) == sub / "weights.pt"


def test_a_newton_schema_on_a_prim_is_not_a_nexus_component(tmp_path):
    """A Newton schema on a prim isn't a nexus component, and the reader skips its attributes.

    Given a prim applying `NewtonPIDControlAPI` and no nexus schema, when resolved, then the build returns
    no component and raises nothing.
    """
    vehicle = tmp_path / "vehicle.usda"
    stage = Usd.Stage.CreateNew(str(vehicle))
    rotor = stage.DefinePrim("/Vehicle/rotor0", "NewtonActuator")
    rotor.ApplyAPI("NewtonPIDControlAPI")
    rotor.GetAttribute("newton:kp").Set(2.0)
    stage.GetRootLayer().Save()

    assert resolve_components(vehicle, ComponentRegistry({})) == []


def test_a_schema_no_class_claims_fails_the_build_and_names_the_prim():
    """A schema no class claims fails the build and names the prim.

    Given the fixture applying a schema no registry entry claims, when resolved, then the build fails
    naming the prim path and the schema.
    """
    with pytest.raises(ValueError) as e:
        resolve_components(FIXTURE, ComponentRegistry({"NexusPx4API": StandIn}))
    assert IMU in str(e.value) and "NexusImuAPI" in str(e.value)


def test_an_authored_attribute_the_schema_does_not_define_fails_the_build_and_names_the_prim(tmp_path):
    """An authored `nexus:` attribute the schema doesn't define fails the build and names the prim.

    Given `nexus:bogus` authored on the fixture's IMU prim, when resolved, then the build fails naming the
    prim path and the attribute.
    """
    vehicle = _fixture_copy(tmp_path)
    stage = Usd.Stage.Open(str(vehicle))
    stage.GetPrimAtPath(IMU).CreateAttribute("nexus:bogus", Sdf.ValueTypeNames.Float, custom=True).Set(1.0)
    stage.GetRootLayer().Save()

    with pytest.raises(ValueError) as e:
        resolve_components(vehicle, ComponentRegistry({"NexusImuAPI": StandIn}))
    assert IMU in str(e.value) and "nexus:bogus" in str(e.value)
