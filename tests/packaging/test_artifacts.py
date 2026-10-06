"""What the wheel and the sdist built from this checkout hold.

The wheel holds the package and the data its code reads, and no agent instruction file. The sdist is a build
input: it holds what builds the wheel, and no agent instruction file and no tests.
"""

import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def dist(tmp_path_factory) -> Path:
    """The folder that `uv build` writes one wheel and one sdist of this checkout into."""
    out = tmp_path_factory.mktemp("dist")
    subprocess.run(["uv", "build", "-o", str(out), str(ROOT)], check=True, capture_output=True)
    return out


@pytest.fixture(scope="module")
def wheel_files(dist) -> set[str]:
    """The path of every file in the wheel."""
    with zipfile.ZipFile(next(dist.glob("*.whl"))) as wheel:
        return {name for name in wheel.namelist() if not name.endswith("/")}


@pytest.fixture(scope="module")
def sdist_files(dist) -> set[str]:
    """The path of every file in the sdist, below its top folder."""
    with tarfile.open(next(dist.glob("*.tar.gz"))) as sdist:
        return {m.name.split("/", 1)[1] for m in sdist.getmembers() if m.isfile()}


def test_the_wheel_holds_no_claude_md(wheel_files):
    """The wheel holds no `CLAUDE.md`."""
    assert sorted(f for f in wheel_files if f.endswith("CLAUDE.md")) == []


def test_the_sdist_holds_no_claude_md(sdist_files):
    """The sdist holds no `CLAUDE.md`."""
    assert sorted(f for f in sdist_files if f.endswith("CLAUDE.md")) == []


def test_the_sdist_holds_no_tests(sdist_files):
    """The sdist holds no tests."""
    assert sorted(f for f in sdist_files if f.startswith("tests/")) == []


def test_the_wheel_holds_every_file_the_code_reads(wheel_files):
    """The wheel still holds every file the code reads."""
    read = {
        "nexus_sim/_src/usd/plugInfo.json",
        "nexus_sim/_src/usd/generatedSchema.usda",
        "nexus_sim/_src/usd/released.json",
        "nexus_sim/_src/config/registry.yaml",
        "nexus_sim/_src/peers/px4_sitl/px4.ref",
        "nexus_sim/_src/peers/px4_sitl/image/Dockerfile",
        "nexus_sim/examples/controllers/px4/box.plan",
        "nexus_sim/_src/peers/kit/peer-src/cesium_globe.py",
        "nexus_sim/_src/peers/kit/peer-src/kit_benchmark.py",
        "nexus_sim/_src/peers/kit/peer-src/link.py",
        "nexus_sim/_src/peers/kit/peer-src/render.py",
        "nexus_sim/_src/peers/kit/peer-src/serve.py",
        "nexus_sim/py.typed",
    }
    assert sorted(read - wheel_files) == []
