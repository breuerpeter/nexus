"""The driver-script public surface: a test script imports these from nexus.

Operators/controllers are *internal* now, see control-surface-api.md: everything reaches the operator
via ``sim.operator`` and the controller via ``sim.controller``; the old ``Pilot``, ``Px4Pilot`` and
``wait_until`` exports on ``na`` no longer exist.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest


def test_top_level_exports_exist():
    import nexus as na

    for name in ("Sim", "BodyState", "JointState", "logger"):
        assert hasattr(na, name), f"nexus.{name} should be a public export"
        assert name in na.__all__, f"nexus.{name} should be listed in __all__"


def test_dropped_pilot_exports_are_gone():
    import nexus as na

    for name in ("Pilot", "Px4Pilot", "wait_until"):
        assert name not in na.__all__, f"nexus.{name} should no longer be a public export"


# The API reference pages: a `::: nexus.<Name>` directive with one dotted segment documents a
# public export; the `_src` directives in operator.md have more segments and stay out.
_API_REFERENCE = Path(__file__).resolve().parents[2] / "docs" / "reference" / "api"
_DIRECTIVE = re.compile(r"^::: nexus\.(\w+)$", re.MULTILINE)
_IMPORT_LINE = re.compile(r"^from nexus import (.+)$", re.MULTILINE)


def test_state_is_no_longer_a_public_export():
    """`nexus.State` is no longer a public export."""
    import nexus as na

    assert "State" not in na.__all__
    with pytest.raises(ImportError):
        from nexus import State  # noqa: F401


def test_api_reference_documents_no_state():
    """The API reference documents no `State`."""
    import nexus as na

    names = []
    for page in sorted(_API_REFERENCE.glob("*.md")):
        names += _DIRECTIVE.findall(page.read_text())
    for line in _IMPORT_LINE.findall((_API_REFERENCE / "core.md").read_text()):
        names += [name.strip() for name in line.split(",")]
    assert names, "no reference directive found"
    assert [name for name in names if name not in na.__all__] == []
    assert "State" not in names


def test_importing_core_loads_neither_newton_nor_warp():
    """Importing core loads neither `newton` nor `warp`."""
    code = "import sys, nexus._src.core; print('newton' in sys.modules, 'warp' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert out.stdout.split() == ["False", "False"]
