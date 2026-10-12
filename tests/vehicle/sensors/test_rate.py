"""The one due-time rule every sensor decimates by, and the check of a declared rate."""

import pytest
import warp as wp

from nexus_sim._src.vehicle.sensors.rate import checked_rate, due


def test_a_sample_whose_due_time_falls_on_a_tick_is_due_at_that_tick_and_not_a_nanosecond_before():
    """A sample whose due time falls on a tick is due at that tick, and not a nanosecond before.

    Given a 24 Hz sensor that has taken 12 samples, the first at 0 s, then its next sample, due at 0.5 s exactly, is
    due 500,000,000 ns after the first and not 499,999,999 ns after it.
    """
    on_tick = due(wp.int64(500_000_000), wp.int32(12), wp.float64(24.0))
    before = due(wp.int64(499_999_999), wp.int32(12), wp.float64(24.0))

    assert (on_tick, before) == (True, False)


def test_a_negative_rate_fails_and_the_message_names_where_the_sensor_is_declared():
    """A negative rate fails, and the message names where the sensor is declared.

    Given the rate -1 for the prim `/vehicle/body/Cam`, then the check raises and the message holds that path.
    """
    with pytest.raises(ValueError, match="/vehicle/body/Cam"):
        checked_rate(-1.0, "/vehicle/body/Cam")
