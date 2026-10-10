"""The driver-script public surface: a test script imports these from nexus_sim.

Guidance and controllers are *internal*: a flight reaches its guidance via ``sim.guidance`` and its
controller through the objects it built, and a PX4 script imports its client from ``nexus_sim.px4``. The old
``Pilot``, ``Px4Pilot`` and ``wait_until`` exports on ``nx`` no longer exist.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


def test_top_level_exports_exist():
    import nexus_sim as nx

    for name in ("Sim", "BodyState", "JointState", "logger"):
        assert hasattr(nx, name), f"nexus_sim.{name} should be a public export"
        assert name in nx.__all__, f"nexus_sim.{name} should be listed in __all__"


def test_dropped_pilot_exports_are_gone():
    import nexus_sim as nx

    for name in ("Pilot", "Px4Pilot", "wait_until"):
        assert name not in nx.__all__, f"nexus_sim.{name} should no longer be a public export"


def test_a_script_imports_the_px4_client_and_the_mission_plan_types_from_nexus_px4():
    """A script imports the PX4 client and the mission plan types from `nexus_sim.px4`, and the top level
    names nothing of PX4.

    Given the package, when a script runs `from nexus_sim.px4 import OffboardClient, MissionItem, Plan,
    read_plan` and imports the mission item constants the PX4 scripts use, then each name resolves to
    its definition in the PX4 Software In The Loop (SITL) peer's folder, and `nexus_sim.__all__` holds no
    PX4 name.
    """
    import nexus_sim as nx
    from nexus_sim.px4 import (
        FRAME_GLOBAL_RELATIVE_ALT,
        NAV_TAKEOFF,
        NAV_WAYPOINT,
        MissionItem,
        OffboardClient,
        Plan,
        read_plan,
    )

    in_peer_folder = all(
        obj.__module__.startswith("nexus_sim._src.peers.px4_sitl.")
        for obj in (OffboardClient, MissionItem, Plan, read_plan)
    )
    # The MAVLink values of the three constants, which PX4 reads in a mission item.
    constants = (NAV_WAYPOINT, NAV_TAKEOFF, FRAME_GLOBAL_RELATIVE_ALT)
    px4_names = [n for n in nx.__all__ if "px4" in n.lower() or n in ("MissionItem", "Plan", "read_plan")]
    assert (in_peer_folder, constants, px4_names) == (True, (16, 22, 3), [])


# The API reference pages: a `::: nexus.<Name>` directive with one dotted segment documents a
# public export; the `_src` directives in guidance.md have more segments and stay out.
_API_REFERENCE = Path(__file__).resolve().parents[2] / "docs" / "reference" / "api"
_DIRECTIVE = re.compile(r"^::: nexus_sim\.(\w+)$", re.MULTILINE)
_IMPORT_LINE = re.compile(r"^from nexus_sim import (.+)$", re.MULTILINE)

# The public surface, pinned: every name `import nexus_sim` exports, sorted. A removed name needs no test of its
# absence: it leaves this list. The run's entry and its helpers, the component contract with its stage, tick,
# signal and sensor-run types, the registry and its registration call, the peer contract, the catalog and the
# launch config, the plant's read types, the signal value types a component reads or writes, and the logger.
PINNED = [
    "BaroSample",
    "BodyState",
    "Catalog",
    "Component",
    "Controls",
    "GpsSample",
    "Image",
    "ImuSample",
    "JointState",
    "LaunchConfig",
    "MagSample",
    "Peer",
    "PointCloud",
    "PoseTwist",
    "PositionGoal",
    "ReferenceTrajectory",
    "Registry",
    "SensorRun",
    "Signal",
    "Sim",
    "SimTime",
    "Stage",
    "Tick",
    "Waypoints",
    "logger",
    "register",
    "save_run_artifacts",
    "sim_argparser",
]


def test_the_public_surface_is_the_pinned_list_and_the_reference_documents_each_name():
    """`import nexus_sim` exports exactly the pinned list, and the API reference documents each name.

    Given a fresh interpreter, when it imports `nexus_sim`, then `__all__` equals the pinned literal, each name
    resolves, and the API reference holds a `::: nexus_sim.<Name>` directive for each.
    """
    code = (
        "import json, nexus_sim as nx\n"
        "print('>', json.dumps([sorted(nx.__all__), [n for n in nx.__all__ if not hasattr(nx, n)]]))"
    )
    out = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
    marked = [line[2:] for line in out.stdout.splitlines() if line.startswith("> ")]
    exported, unresolved = json.loads(marked[0]) if marked else ([], ["no output"])
    documented = set()
    for page in sorted(_API_REFERENCE.glob("*.md")):
        documented |= set(_DIRECTIVE.findall(page.read_text()))

    assert (exported, unresolved, [n for n in PINNED if n not in documented]) == (PINNED, [], []), out.stderr


def test_state_is_no_longer_a_public_export():
    """`nexus_sim.State` is no longer a public export."""
    import nexus_sim as nx

    assert "State" not in nx.__all__
    with pytest.raises(ImportError):
        from nexus_sim import State  # noqa: F401


def test_api_reference_documents_no_state():
    """The API reference documents no `State`."""
    import nexus_sim as nx

    names = []
    for page in sorted(_API_REFERENCE.glob("*.md")):
        names += _DIRECTIVE.findall(page.read_text())
    for line in _IMPORT_LINE.findall((_API_REFERENCE / "core.md").read_text()):
        names += [name.strip() for name in line.split(",")]
    assert names, "no reference directive found"
    assert [name for name in names if name not in nx.__all__] == []
    assert "State" not in names


def test_nexus_catalog_loads_the_bundled_catalog():
    """`nexus_sim.Catalog` names the vehicle and scene catalog.

    Given a fresh interpreter, when it imports `nexus_sim`, then `nexus_sim.Catalog` loads the bundled catalog
    with its two vehicles.
    """
    code = "import nexus_sim as nx; print(sorted(nx.Catalog.from_yaml().vehicles))"
    out = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
    assert out.stdout.strip() == "['astro_max_base', 'astro_max_fpv']", out.stderr


def test_importing_core_loads_no_newton():
    """Importing core loads no `newton`: core runs on Warp, never on a physics backend."""
    code = "import sys, nexus_sim._src.core; print('newton' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert out.stdout.split() == ["False"]


def test_envsample_and_environment_are_no_longer_public_exports():
    """`EnvSample` and `Environment` leave the public surface and the API reference."""
    import nexus_sim as nx

    assert "EnvSample" not in nx.__all__ and "Environment" not in nx.__all__
    with pytest.raises(ImportError):
        from nexus_sim import EnvSample  # noqa: F401


def test_api_reference_documents_no_envsample():
    """`EnvSample` and `Environment` leave the public surface and the API reference."""
    names = []
    for page in sorted(_API_REFERENCE.glob("*.md")):
        names += _DIRECTIVE.findall(page.read_text())
    for line in _IMPORT_LINE.findall((_API_REFERENCE / "core.md").read_text()):
        names += [name.strip() for name in line.split(",")]
    assert "EnvSample" not in names and "Environment" not in names


def test_the_public_surface_names_no_measurement():
    """The public surface names no `Measurement`.

    Given a fresh interpreter, when it imports `nexus_sim`, then `Measurement` isn't in `nexus_sim.__all__`, and
    `nexus_sim.Measurement` raises `AttributeError`.
    """
    code = (
        "import nexus_sim as nx\n"
        "try:\n"
        "    nx.Measurement\n"
        "    raised = False\n"
        "except AttributeError:\n"
        "    raised = True\n"
        "print('Measurement' in nx.__all__, raised)"
    )
    out = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
    assert out.stdout.strip() == "False True", out.stderr
