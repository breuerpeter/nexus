"""The renderer the RTX sensors use ships as an extra: the core dependencies leave it out, and the `rtx` extra pins it."""

import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _name(entry: str) -> str:
    """The distribution a dependency entry names, without its version or markers."""
    return re.split(r"[\s\[<>=!~;]", entry, maxsplit=1)[0].lower()


def test_the_core_names_no_renderer_and_the_rtx_extra_pins_ovrtx_and_ovstage_exactly():
    """The core dependencies name neither `ovrtx` nor `ovstage`, and the `rtx` extra pins each to one exact version.

    Given `pyproject.toml`, when the test reads its dependencies, then the core list names neither, and the `rtx`
    extra holds both, each pinned with `==`.
    """
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    core = {_name(r) for r in project["dependencies"]}
    rtx = {_name(r): "==" in r for r in project.get("optional-dependencies", {}).get("rtx", [])}

    assert (core & {"ovrtx", "ovstage"}, {k: rtx.get(k) for k in ("ovrtx", "ovstage")}) == (
        set(),
        {"ovrtx": True, "ovstage": True},
    )
