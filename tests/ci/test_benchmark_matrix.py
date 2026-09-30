"""The real-time factor benchmark matrix, scripts/ci/benchmark_matrix.py: the cells it plans to fly, read from its
`--list` output as the GPU workflow reads them.
"""

import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _rtx_scenes() -> list[str]:
    # A token so the plan keeps the cesium cell instead of skipping it.
    env = {**os.environ, "CESIUM_ION_TOKEN": "test"}
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ci" / "benchmark_matrix.py"), "--list", "--only", "rtx"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [c["scene"] for c in json.loads(out)["cells"]]


def test_the_rtx_row_flies_a_static_geometry_scene_as_its_middle_column():
    """The isaacsim matrix's RTX row flies the scene as its middle column: `empty`, then
    `powerline`, then `cesium`.
    """
    assert _rtx_scenes() == ["empty", "powerline", "cesium"]
