"""The driver-script public surface: a test script imports these from nexus.

Guidance and controllers are *internal*: a flight reaches its guidance via ``sim.guidance`` and its
controller via ``sim.controller``, and a PX4 script imports its client from ``nexus.px4``. The old
``Pilot``, ``Px4Pilot`` and ``wait_until`` exports on ``na`` no longer exist.
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


def test_a_script_imports_the_px4_client_and_the_mission_plan_types_from_nexus_px4():
    """A script imports the PX4 client and the mission plan types from `nexus.px4`, and the top level
    names nothing of PX4.

    Given the package, when a script runs `from nexus.px4 import OffboardClient, MissionItem, Plan,
    read_plan` and imports the mission item constants the PX4 scripts use, then each name resolves to
    its definition in the PX4 Software In The Loop (SITL) peer's folder, and `nexus.__all__` holds no
    PX4 name.
    """
    import nexus as na
    from nexus.px4 import (
        FRAME_GLOBAL_RELATIVE_ALT,
        NAV_TAKEOFF,
        NAV_WAYPOINT,
        MissionItem,
        OffboardClient,
        Plan,
        read_plan,
    )

    in_peer_folder = all(
        obj.__module__.startswith("nexus._src.peers.px4_sitl.")
        for obj in (OffboardClient, MissionItem, Plan, read_plan)
    )
    # The MAVLink values of the three constants, which PX4 reads in a mission item.
    constants = (NAV_WAYPOINT, NAV_TAKEOFF, FRAME_GLOBAL_RELATIVE_ALT)
    px4_names = [n for n in na.__all__ if "px4" in n.lower() or n in ("MissionItem", "Plan", "read_plan")]
    assert (in_peer_folder, constants, px4_names) == (True, (16, 22, 3), [])


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


def test_envsample_and_environment_are_no_longer_public_exports():
    """`EnvSample` and `Environment` leave the public surface and the API reference."""
    import nexus as na

    assert "EnvSample" not in na.__all__ and "Environment" not in na.__all__
    with pytest.raises(ImportError):
        from nexus import EnvSample  # noqa: F401


def test_api_reference_documents_no_envsample():
    """`EnvSample` and `Environment` leave the public surface and the API reference."""
    names = []
    for page in sorted(_API_REFERENCE.glob("*.md")):
        names += _DIRECTIVE.findall(page.read_text())
    for line in _IMPORT_LINE.findall((_API_REFERENCE / "core.md").read_text()):
        names += [name.strip() for name in line.split(",")]
    assert "EnvSample" not in names and "Environment" not in names
