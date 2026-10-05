"""The nexus Universal Scene Description (USD) schema plugin: in the wheel, registered on import, applied in full."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).with_name("conformance.usda")


def test_the_plugin_ships_in_the_wheel_and_import_nexus_registers_it(tmp_path):
    """The plugin ships in the wheel and `import nexus` registers it, so the Inertial Measurement Unit (IMU) schema needs no other call.

    Given the wheel built from the checkout installed into a folder of its own, when `import nexus` runs
    from there, then `Usd.SchemaRegistry` returns the IMU schema's applied API definition with each
    attribute and its fallback. The folder stands in for a fresh venv: the wheel's own files answer the
    import, and only its dependencies come from this environment.
    """
    wheels = tmp_path / "wheels"
    subprocess.run(["uv", "build", "--wheel", "-o", str(wheels), str(ROOT)], check=True, capture_output=True)
    site = tmp_path / "site"
    wheel = next(wheels.glob("*.whl"))
    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable, "--no-deps", "--target", str(site), str(wheel)],
        check=True,
        capture_output=True,
    )
    code = (
        "import nexus\n"
        f"assert nexus.__file__.startswith({str(site)!r}), nexus.__file__\n"
        "from pxr import Usd\n"
        "d = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition('NexusImuAPI')\n"
        "names = sorted(d.GetPropertyNames())\n"
        "print(names, [round(d.GetAttributeFallbackValue(n), 6) for n in names])\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(site)},
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "['nexus:accNoise', 'nexus:gyroNoise', 'nexus:rate'] [0.02, 0.02, 0.0]"


def test_the_imu_schema_defines_its_two_noise_attributes_and_the_rate():
    """The Inertial Measurement Unit (IMU) schema defines `nexus:accNoise`, `nexus:gyroNoise` and the shared `nexus:rate`."""
    from pxr import Usd

    import nexus  # noqa: F401  # registers the plugin

    definition = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition("NexusImuAPI")
    assert sorted(definition.GetPropertyNames()) == ["nexus:accNoise", "nexus:gyroNoise", "nexus:rate"]


def test_every_sensor_schema_shares_one_rate_attribute():
    """Every sensor schema shares one rate attribute, and each sensor receives its authored rate.

    Given the plugin, when a test lists each sensor schema's attributes, then all seven define
    `nexus:rate` with one type and one unit.
    """
    from pxr import Usd

    import nexus  # noqa: F401  # registers the plugin

    sensors = ("Imu", "Mag", "Baro", "Gps", "Camera", "ThermalCamera", "Lidar")
    rates = set()
    for sensor in sensors:
        definition = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(f"Nexus{sensor}API")
        rate = definition.GetAttributeDefinition("nexus:rate")
        (unit,) = [line.strip() for line in rate.GetDocumentation().splitlines() if line.strip().startswith("Units:")]
        rates.add((str(rate.GetTypeName()), unit))

    assert rates == {("float", "Units: hertz")}


def test_the_conformance_fixture_applies_every_schema_the_plugin_defines():
    """The conformance fixture applies every schema the plugin defines.

    Given the plugin and the fixture, when a test lists the schemas each defines and applies, then the two
    sets are equal.
    """
    from pxr import Usd

    import nexus  # noqa: F401  # registers the plugin
    from nexus._src.usd import schema_names

    stage = Usd.Stage.Open(str(FIXTURE))
    applied = {name for prim in stage.Traverse() for name in prim.GetAppliedSchemas()}
    sensors = {f"Nexus{name}API" for name in ("Imu", "Mag", "Baro", "Gps", "Camera", "ThermalCamera", "Lidar")}
    assert applied == set(schema_names()) == sensors | {"NexusPx4API", "NexusPx4SitlAPI"}


def test_the_script_beside_the_fixture_reproduces_it(tmp_path):
    """The script beside the fixture reproduces it.

    Given the script, when it runs, then its output matches the committed fixture byte for byte.
    """
    out = tmp_path / "conformance.usda"
    script = Path(__file__).with_name("author_conformance.py")
    r = subprocess.run([sys.executable, str(script), str(out)], check=False, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.read_bytes() == FIXTURE.read_bytes()
