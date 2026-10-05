"""A run's components of one role carry distinct names, since a recorded row's path ends in the name.

Real end-to-end on the Warp CPU backend against the Kit peer's fake. Skipped if rerun or newton are
missing.
"""

import pytest

pytest.importorskip("rerun")
pytest.importorskip("newton")

from tests.conftest import fly_recorded

# Two cameras whose prims are both named `cam`: one on the base body, one on a rotor's body.
TWO_CAMERAS_NAMED_CAM = """\
over "astro_max"
{
    over "Geometry"
    {
        over "body_frd"
        {
            def Camera "cam"
            {
                int sensor:width = 64
                int sensor:height = 48
                double sensor:rate_hz = 25
            }

            over "rotor_1"
            {
                def Camera "cam"
                {
                    int sensor:width = 64
                    int sensor:height = 48
                    double sensor:rate_hz = 25
                }
            }
        }
    }
}
"""


def test_two_components_of_one_role_whose_prims_share_a_name_fail_the_build_and_the_error_names_both_prims(tmp_path):
    """Two components of one role whose prims share a name fail the build, and the error names both prims.

    Given a vehicle with two cameras whose prims are both named `cam`, under different bodies, when
    the run builds, then the build fails and the error names both prim paths.
    """
    message = "the build did not fail"
    try:
        fly_recorded(tmp_path, "astro_max_base", layer=TWO_CAMERAS_NAMED_CAM, ticks=1)
    except ValueError as exc:
        message = str(exc)

    assert (
        "/astro_max/Geometry/body_frd/cam" in message,
        "/astro_max/Geometry/body_frd/rotor_1/cam" in message,
    ) == (True, True), message
