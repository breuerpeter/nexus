"""The vehicle and scene catalog a run flies: the one the wheel ships, and a project's beside it."""

from nexus_sim import LaunchConfig
from nexus_sim._src.config import resolve

PROJECT = """\
vehicles:
  project_vehicle:
    usd: { url: "file:///project_vehicle.usda", sha256: "0" }
"""


def test_a_run_inside_a_project_flies_the_nearest_catalog_beside_the_bundled_one(tmp_path, monkeypatch):
    """A run started inside a project flies the vehicles of the nearest `nexus.catalog.yaml` in its
    working directory or a directory over it, beside the bundled ones.

    Given a project directory whose `nexus.catalog.yaml` lists a vehicle at a local Universal Scene
    Description (USD) path, and a subdirectory of it as the working directory, when a run resolves
    that vehicle and `astro_max_base`, then both resolve, the first from the project's file.
    """
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "nexus.catalog.yaml").write_text(PROJECT)
    monkeypatch.chdir(project / "scripts")

    own = resolve(LaunchConfig(vehicle="project_vehicle", scene="empty"), fetch=False)
    bundled = resolve(LaunchConfig(vehicle="astro_max_base", scene="empty"), fetch=False)

    assert (own.tested_config.vehicle_usd.url, bundled.tested_config.vehicle) == (
        "file:///project_vehicle.usda",
        "astro_max_base",
    )
