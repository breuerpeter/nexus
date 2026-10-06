"""A run's components of one role carry distinct names, since a recorded row's path ends in the name.

Real end-to-end on the Warp CPU backend of the local fixture vehicle in ``tests/usd/sensor_vehicle.py``,
against the Kit peer's fake. Skipped if rerun or newton are
missing.
"""

import pytest

pytest.importorskip("rerun")
pytest.importorskip("newton")

from tests.conftest import fly_recorded
from tests.usd import sensor_vehicle as sv

# A camera prim named `cam`, 64 by 48 at 25 Hz.
CAM = sv.prim(
    "cam", "NexusCameraAPI", "int nexus:width = 64\nint nexus:height = 48\nfloat nexus:rate = 25", kind="Camera"
)
# Two cameras whose prims are both named `cam`: one on the base body, one on a rotor's body.
TWO_CAMERAS_NAMED_CAM = CAM + 'over "rotor_1"\n{\n' + CAM + "}\n"


def test_two_components_of_one_role_whose_prims_share_a_name_fail_the_build_and_the_error_names_both_prims(tmp_path):
    """Two components of one role whose prims share a name fail the build, and the error names both prims.

    Given a vehicle with two cameras whose prims are both named `cam`, under different bodies, when
    the run builds, then the build fails and the error names both prim paths.
    """
    message = "the build did not fail"
    try:
        fly_recorded(tmp_path, sv.vehicle(tmp_path, TWO_CAMERAS_NAMED_CAM), ticks=1)
    except ValueError as exc:
        message = str(exc)

    assert (
        f"{sv.BODY}/cam" in message,
        f"{sv.BODY}/rotor_1/cam" in message,
    ) == (True, True), message
