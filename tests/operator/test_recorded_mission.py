"""Where an operator's mission sits in a run's recording.

The test reads the session's one recorded flight, which ``tests/conftest.py`` flies under an
operator that plans a two-waypoint mission. Skipped if rerun or newton are missing.
"""

import pytest

pytest.importorskip("rerun")
pytest.importorskip("newton")


def test_the_operators_waypoints_and_reference_sit_under_sim_operator(recorded_flight, rrd_rows):
    """The operator's waypoints and reference sit under `sim/operator/`.

    Given a recorded flight with an operator that holds a two-waypoint mission, when it ends, then the
    `.rrd` holds `/sim/operator/waypoints/wp_0`, `/sim/operator/waypoints/wp_1` and `/sim/operator/reference`.
    """
    mission = sorted({entity for entity, _, _ in rrd_rows(recorded_flight) if "/operator/" in entity})

    assert mission == ["/sim/operator/reference", "/sim/operator/waypoints/wp_0", "/sim/operator/waypoints/wp_1"]
