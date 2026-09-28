"""What a project that installs the package resolves from the wheel's metadata."""

import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _locked(name: str) -> str:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    return next(p["version"] for p in lock["package"] if p["name"] == name)


def test_an_installed_package_resolves_the_warp_the_lock_pins(tmp_path):
    """A project that installs `nexus-sim` resolves the `warp-lang` version the checkout's `uv.lock` pins.

    The lock and `[tool.uv.sources]` never reach a consumer, so the test resolves a wheel built from
    this checkout the way a consumer does: from a folder outside the checkout, against PyPI alone, with
    no project config.
    """
    wheels = tmp_path / "wheels"
    subprocess.run(["uv", "build", "--wheel", "-o", str(wheels), str(ROOT)], check=True, capture_output=True)
    wheel = next(wheels.glob("*.whl"))
    (tmp_path / "requirements.in").write_text(f"nexus-sim @ {wheel.as_uri()}\n")

    out = subprocess.run(
        [
            "uv", "pip", "compile", "requirements.in", "--no-config", "--no-header",
            "--python-version", "3.12", "--python-platform", "x86_64-manylinux_2_28",
            "--default-index", "https://pypi.org/simple",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )  # fmt: skip

    resolved = re.search(r"^warp-lang==(\S+)$", out.stdout, re.MULTILINE)
    assert resolved and resolved.group(1) == _locked("warp-lang")
