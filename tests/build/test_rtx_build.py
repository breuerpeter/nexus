"""What the build of a run does with a vehicle's RTX sensors before a frame renders: it refuses a lidar the renderer
can't scan and an environment that lacks the renderer, and a vehicle with no RTX sensor loads no renderer.

Each test builds the fixture vehicle on the Warp CPU backend, flown by the stand-in controller, and the docker client
fails a test that asks it for a container. Skipped without newton or pxr.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

from tests.usd import sensor_vehicle as sv

pytestmark = pytest.mark.usefixtures("warp_cpu", "no_docker")

ROOT = Path(__file__).resolve().parents[2]
CAMERA = sv.prim(
    "Cam", "NexusCameraAPI", "int nexus:width = 8\nint nexus:height = 6\nfloat nexus:rate = 25", kind="Camera"
)


def _build_error(vehicle: str) -> str:
    """The message of the error a build of the fixture vehicle at `vehicle` raises, or "" when it builds."""
    try:
        sv.build(vehicle).close()
    except Exception as exc:
        return str(exc)
    return ""


def test_a_lidar_prim_without_the_generic_lidar_schema_fails_the_build_naming_the_prim_and_the_schema(tmp_path):
    """A lidar prim that lacks `OmniSensorGenericLidarCoreAPI` fails the build with an error naming the prim and the
    schema, not a run of empty scans.

    Given the fixture vehicle with a lidar at 10 Hz on an `OmniLidar` prim `/vehicle/body/Scan` that applies
    `NexusLidarAPI` alone, when the run builds, then the build raises, and the message names the prim and
    `OmniSensorGenericLidarCoreAPI`.
    """
    lidar = sv.prim("Scan", "NexusLidarAPI", "float nexus:rate = 10", kind="OmniLidar")
    message = _build_error(sv.vehicle(tmp_path, lidar))

    assert ("/vehicle/body/Scan" in message, "OmniSensorGenericLidarCoreAPI" in message) == (True, True), message


def test_an_rtx_vehicle_without_the_rtx_extra_fails_the_build_naming_the_extra(tmp_path, monkeypatch):
    """A vehicle with an RTX sensor, in an environment without the `rtx` extra, fails the build with an error naming
    `nexus-uav[rtx]`.

    Given the fixture vehicle with a camera, in a process where `ovrtx` and `ovstage` don't import, when the run
    builds, then the build raises, and the message names `nexus-uav[rtx]`.
    """
    monkeypatch.setitem(sys.modules, "ovrtx", None)  # an import of a module set to None raises ImportError
    monkeypatch.setitem(sys.modules, "ovstage", None)
    message = _build_error(sv.vehicle(tmp_path, CAMERA))

    assert "nexus-uav[rtx]" in message, message


# A run of the fixture vehicle with no RTX sensor, in a process of its own, which prints the renderer's modules it
# loaded.
_RUN = """
import json, sys
import warp as wp
import tests.usd.stand_in.nexus_stand_in
from tests.usd import sensor_vehicle as sv
with wp.ScopedDevice("cpu"):
    sv.steps(sv.build(sys.argv[1]), 5)
print(json.dumps(sorted(m for m in ("ovrtx", "ovstage") if m in sys.modules)))
"""


def test_a_run_of_a_vehicle_with_no_rtx_sensor_imports_neither_ovrtx_nor_ovstage(tmp_path):
    """A run of a vehicle with no RTX sensor imports neither `ovrtx` nor `ovstage`.

    Given the fixture vehicle with no sensor, when a process of its own builds it and steps 5 ticks, then neither
    `ovrtx` nor `ovstage` is among the process's modules.
    """
    vehicle = sv.vehicle(tmp_path)
    out = subprocess.run(
        [sys.executable, "-c", _RUN, vehicle], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout  # fmt: skip

    assert json.loads(out.strip().splitlines()[-1]) == []
