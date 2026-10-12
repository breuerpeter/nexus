"""The world's values live in Newton's model: the origin a scene's ``PhysicsScene`` authors, its magnetic field,
derived from the origin where the scene authors none, its pressure at sea level and its temperature. Skipped
without newton or pxr.
"""

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

import newton

from nexus_sim._src.scene import world

pytestmark = pytest.mark.usefixtures("warp_cpu")

ZURICH = (47.3769, 8.5417, 408.0)
# The World Magnetic Model (WMM) field at Zurich as a North East Down (NED) vector in gauss, from PX4's
# coarse table: declination 2.6°, inclination 63.2°, strength 0.481, in the world's axes X north, Y west, Z up.
ZURICH_WORLD = (0.216, -0.010, -0.429)
AUTHORED_FIELD = (0.2, -0.05, -0.45)


def _scene(tmp_path, values: str) -> str:
    """A scene whose ``PhysicsScene`` applies ``NexusSceneAPI`` and authors `values`, one attribute per line."""
    path = tmp_path / "scene.usda"
    body = "".join(f"        {line.strip()}\n" for line in values.strip().splitlines())
    path.write_text(
        '#usda 1.0\n(\n    defaultPrim = "World"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\n'
        'def Xform "World"\n{\n    def PhysicsScene "PhysicsScene" (\n        prepend apiSchemas = ["NexusSceneAPI"]\n    )\n'
        f"    {{\n{body}    }}\n}}\n"
    )
    return str(path)


def _model(scene: str):
    """The model of `scene`, built as the physics builds one: the values declared first and derived last."""
    builder = newton.ModelBuilder()
    world.declare(builder)
    builder.add_usd(scene)
    world.derive(builder)
    return builder.finalize()


def test_the_field_derives_from_the_origin_where_the_scene_authors_none(tmp_path):
    """A scene that authors its origin and no field gives the model the WMM field at the origin, in world axes.

    Given a scene whose `PhysicsScene` authors Zurich as its origin and no field, when the model builds, then
    its field is Zurich's WMM field as (n, -e, -d).
    """
    model = _model(_scene(tmp_path, f"double3 nexus:geodeticOrigin = {ZURICH}"))

    np.testing.assert_allclose(model.nexus.magnetic_field.numpy()[0], ZURICH_WORLD, atol=1e-3)


def test_the_values_a_scene_authors_reach_the_model_as_authored(tmp_path):
    """The origin, the field, the pressure and the temperature a scene authors reach the model as authored.

    Given a scene whose `PhysicsScene` authors all four, when the model builds, then each of the model's world
    values is the authored one, and the field isn't derived.
    """
    model = _model(
        _scene(
            tmp_path,
            f"double3 nexus:geodeticOrigin = {ZURICH}\nfloat3 nexus:magneticField = {AUTHORED_FIELD}\n"
            "float nexus:seaLevelPressure = 950\nfloat nexus:temperature = 31",
        )
    )

    read = (
        *model.nexus.geodetic_origin.numpy()[0],
        *model.nexus.magnetic_field.numpy()[0],
        float(model.nexus.sea_level_pressure.numpy()[0]),
        float(model.nexus.temperature.numpy()[0]),
    )
    assert read == pytest.approx((*ZURICH, *AUTHORED_FIELD, 950.0, 31.0), abs=1e-5)


def test_a_scene_that_authors_no_pressure_or_temperature_gives_the_standard_atmosphere(tmp_path):
    """A scene that authors neither its pressure nor its temperature gives the model 1013.25 hPa and 25 °C."""
    model = _model(_scene(tmp_path, f"double3 nexus:geodeticOrigin = {ZURICH}"))

    assert (float(model.nexus.sea_level_pressure.numpy()[0]), float(model.nexus.temperature.numpy()[0])) == (
        1013.25,
        25.0,
    )
