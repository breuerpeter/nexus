"""An example's run clock times its run and the part Warp spends compiling kernels, so a time gate can leave out
the one-time compiles a cold kernel cache costs. Warp reports each module it loads through its logger.
"""

import warp as wp

from nexus_uav.examples._lib.eval_dump import RunClock


def test_the_run_clock_counts_the_kernel_compiles_warp_reports_and_no_cached_load():
    """The run clock counts the kernel compiles Warp reports and no load from Warp's cache.

    Given a run during which Warp reports one module compiled in 1500 ms and one loaded from its cache in 20 ms,
    when the run's clock reads the reports, then its compile time is 1.5 s.
    """
    with RunClock() as clock:
        wp.get_logger().info("Module a 1234567 load on device 'cuda:0' took 1500.00 ms  (compiled)")
        wp.get_logger().info("Module b 7654321 load on device 'cuda:0' took 20.00 ms  (cached)")

    assert clock.stats()["compile_s"] == 1.5


def test_the_run_clock_passes_every_report_on_to_the_logger_it_replaced(capsys):
    """The run clock passes every report on to the logger it replaced.

    Given Warp's default logger, when Warp reports a module load during a timed run, then the report still
    reaches the run's output.
    """
    with RunClock():
        wp.get_logger().info("Module a 1234567 load on device 'cpu' took 3.00 ms  (cached)")

    assert "Module a 1234567 load on device 'cpu' took 3.00 ms  (cached)" in capsys.readouterr().out
