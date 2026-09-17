"""The ``px4_sitl`` example's warnings gate: what it counts as a sim problem, and what it forgives."""

from nexus.examples.controllers.px4.log_warnings import px4_warnings

# One PX4 boot log: two lines that must gate, both allowlist entries, an info-level line, and a warning
# with terminal color escapes written on a \r-terminated line; PX4 writes both.
LOG = (
    "INFO  [px4] startup\n"
    "WARN  [health] Preflight Fail: Strong magnetic interference\n"
    "WARN  [param] Parameter EKF2_MAG_TYPE not found\n"
    "\x1b[31mERROR\x1b[0m [simulator_mavlink] poll timeout 0, 22\r"
    "WARN  [commander] Preflight Fail: no heading reference\r\n"
    "INFO  [commander] Ready for takeoff\n"
)


def _write(tmp_path, text):
    p = tmp_path / "px4.log"
    p.write_text(text)
    return p


def test_flags_only_sim_problems(tmp_path):
    assert px4_warnings(_write(tmp_path, LOG)) == [
        "WARN  [health] Preflight Fail: Strong magnetic interference",
        "ERROR [simulator_mavlink] poll timeout 0, 22",
    ]


def test_carriage_return_breaks_lines(tmp_path):
    r"""PX4 ends lines with a bare \r; without the break the whole log reads as one line and a
    single allowlisted token would suppress every warning on it.
    """
    log = "WARN  [param] benign\rWARN  [health] Preflight Fail: Strong magnetic interference\r"
    assert px4_warnings(_write(tmp_path, log)) == [
        "WARN  [health] Preflight Fail: Strong magnetic interference",
    ]


def test_missing_file_is_not_a_warning(tmp_path):
    assert px4_warnings(tmp_path / "never-written.log") == []


def test_logger_capacity_overflow_is_forgiven_per_topic(tmp_path):
    """The pinned PX4 tree's logger overflows its subscription cap on four known topics; a
    different topic overflowing still gates.
    """
    log = (
        "WARN  [logger] Too many subscriptions, failed to add: obstacle_distance 0\n"
        "WARN  [logger] Too many subscriptions, failed to add: vehicle_mocap_odometry 0\n"
        "WARN  [logger] Too many subscriptions, failed to add: vehicle_attitude 0\n"
    )
    assert px4_warnings(_write(tmp_path, log)) == [
        "WARN  [logger] Too many subscriptions, failed to add: vehicle_attitude 0",
    ]
