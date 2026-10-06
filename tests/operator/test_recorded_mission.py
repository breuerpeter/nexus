"""Where a guidance's mission sits in a run's recording.

The test reads the session's one recorded flight, which ``tests/conftest.py`` flies under a
guidance that plans a two-waypoint mission. Skipped if rerun or newton are missing.
"""

import pytest

pytest.importorskip("rerun")
pytest.importorskip("newton")


def test_the_guidances_waypoints_and_reference_sit_under_sim_guidance(recorded_flight, rrd_rows):
    """The guidance's waypoints and reference sit under `sim/guidance/`.

    Given a recorded flight with a guidance that holds a two-waypoint mission, when it ends, then the
    `.rrd` holds `/sim/guidance/waypoints/wp_0`, `/sim/guidance/waypoints/wp_1` and `/sim/guidance/reference`.
    """
    mission = sorted({entity for entity, _, _ in rrd_rows(recorded_flight) if "/guidance/" in entity})

    assert mission == ["/sim/guidance/reference", "/sim/guidance/waypoints/wp_0", "/sim/guidance/waypoints/wp_1"]
