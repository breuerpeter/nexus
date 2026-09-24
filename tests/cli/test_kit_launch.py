"""The Kit launch from the host, tested through the command-line tool a caller installs.

Each test runs the tool in a child process from a folder outside the checkout, with ``DOCKER_HOST``
pointing at a socket that doesn't exist, so no container starts and the tests need no GPU.
"""

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# A vehicle Universal Scene Description (USD) file with one camera prim: an RTX sensor, so `--runtime auto` picks the Kit runtime.
CAMERA_USD = '#usda 1.0\n(\n    defaultPrim = "vehicle"\n)\ndef Xform "vehicle" {\n    def Camera "cam" {}\n}\n'
PLAIN_USD = '#usda 1.0\n(\n    defaultPrim = "world"\n)\ndef Xform "world" {}\n'


def _ref(path: Path) -> str:
    return f'{{ url: "{path.as_uri()}", sha256: {hashlib.sha256(path.read_bytes()).hexdigest()} }}'


def _env(tmp_path: Path) -> dict:
    """No reachable daemon, and every cache in the test's own folder."""
    return {
        **os.environ,
        "DOCKER_HOST": f"unix://{tmp_path / 'no-daemon.sock'}",
        "HOME": str(tmp_path / "home"),
        "NEXUS_ASSET_CACHE": str(tmp_path / "cache"),
    }


def test_the_host_fetches_the_vehicle_and_the_scene_before_the_kit_container_starts(tmp_path):
    """Before the Kit container starts, the host fetches the vehicle and the scene the run renders.

    The Kit container reads the run's assets from the host's cache and has no credentials for an
    `s3://` catalog, so what it renders must already be in that cache. The vehicle carries a camera
    prim, so the run needs Kit.
    """
    vehicle = tmp_path / "assets" / "v.usda"
    scene = tmp_path / "assets" / "s.usda"
    vehicle.parent.mkdir()
    vehicle.write_text(CAMERA_USD)
    scene.write_text(PLAIN_USD)
    registry = tmp_path / "catalog.yaml"
    registry.write_text(
        f"vehicles:\n  - name: v\n    usd: {_ref(vehicle)}\n"
        f"scenes:\n  s:\n    usd: {_ref(scene)}\n    geodetic_origin: null\n"
    )
    exe = shutil.which("nexus")
    assert exe, "no `nexus` command on the path"

    subprocess.run(
        [exe, "run", "--vehicle", "v", "--scene", "s", "--registry", str(registry)],
        cwd=tmp_path,
        env=_env(tmp_path),
        check=False,
        capture_output=True,
        timeout=300,
    )

    cached = {p.relative_to(tmp_path / "cache") for p in (tmp_path / "cache").rglob("*") if p.is_file()}
    expected = {
        Path(hashlib.sha256(vehicle.read_bytes()).hexdigest()) / "v.usda",
        Path(hashlib.sha256(scene.read_bytes()).hexdigest()) / "s.usda",
    }
    assert cached == expected


def test_an_rtx_run_without_docker_names_docker_and_no_missing_file(tmp_path):
    """With Docker missing or unreachable, an RTX run from an installed package fails with a message
    that names Docker and names no file that doesn't exist.

    The package runs from a wheel built from this checkout, installed into a folder of its own that
    comes first on the path, so the tool runs from site-packages as a consumer's does, not from the
    checkout.
    """
    pytest.importorskip("pxr")
    wheels = tmp_path / "wheels"
    site = tmp_path / "site"
    subprocess.run(["uv", "build", "--wheel", "-o", str(wheels), str(ROOT)], check=True, capture_output=True)
    wheel = next(wheels.glob("*.whl"))
    subprocess.run(
        ["uv", "pip", "install", "--no-deps", "--target", str(site), "--python", sys.executable, str(wheel)],
        check=True,
        capture_output=True,
    )
    consumer = tmp_path / "consumer"
    consumer.mkdir()
    vehicle = consumer / "cam.usda"
    vehicle.write_text(CAMERA_USD)
    # A catalog with an empty default scene, so the run fetches nothing over the network.
    (consumer / "nexus.registry.yaml").write_text("scenes:\n  empty: {}\n")
    env = {**_env(tmp_path), "PYTHONPATH": str(site)}

    out = subprocess.run(
        [sys.executable, "-c", "from nexus._src.cli.main import main; main()", "run", "--vehicle", str(vehicle)],
        cwd=consumer,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )

    message = (out.stdout + out.stderr).replace(env["DOCKER_HOST"], "")
    # A path to a file has a suffix. A prim path such as `/vehicle/cam`, which the log also prints, has none.
    paths = {m.rstrip(".)]") for m in re.findall(r"(?<![\w:/])/[^\s'\"`:,]+", message)}
    missing = sorted(p for p in paths if Path(p).suffix and not os.path.exists(p))
    assert (out.returncode != 0, "docker" in message.lower(), missing) == (True, True, [])
