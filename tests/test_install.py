"""What a project that installs the package resolves from the wheel's metadata."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_an_installed_package_resolves_a_warp_below_the_first_broken_one(tmp_path):
    """A project that installs `nexus-sim` resolves a `warp-lang` below 1.16.0, the first version whose
    physics build fails.

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
    assert resolved and tuple(map(int, resolved.group(1).split(".")[:2])) < (1, 16)
