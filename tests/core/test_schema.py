"""The tick's sim time: whole nanoseconds, which a stage reads in seconds and a ``HIL_*`` message in microseconds."""

from nexus_sim._src.core.schema import SimTime


def test_a_sim_time_holds_whole_nanoseconds_and_reads_them_in_seconds_and_microseconds():
    """A sim time holds whole nanoseconds, and reads them in seconds and in microseconds.

    Given the sim time of the tenth 4 ms tick, 40,000,000 ns, then it reads 0.04 s exactly, not the float sum's
    0.04000000000000001, and 40,000 µs.
    """
    t = SimTime(40_000_000, 10)

    assert (t.time_ns, t.sim_time, t.time_usec) == (40_000_000, 0.04, 40_000)
