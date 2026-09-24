"""The ``px4_sitl`` example's warnings gate: what it counts as a sim problem, and what it forgives."""

import re

from nexus.examples.controllers.px4.log_warnings import WARN_ALLOWLIST, px4_warnings

# The airframe's boot-time line for a parameter this PX4 build lacks, as PX4's `param` command prints it.
ABSENT_PARAM = "ERROR [param] Parameter MAV_0_CONFIG not found."

# One PX4 boot log: two lines that must gate, both allowlist entries, an info-level line, and a warning
# with terminal color escapes written on a \r-terminated line; PX4 writes both.
LOG = (
    "INFO  [px4] startup\n"
    "WARN  [health] Preflight Fail: Strong magnetic interference\n"
    f"{ABSENT_PARAM}\n"
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
    log = f"{ABSENT_PARAM}\rWARN  [health] Preflight Fail: Strong magnetic interference\r"
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


def test_allowlist_names_messages_never_modules():
    """The allowlist names messages, never modules: each ``WARN_ALLOWLIST`` entry is the text of
    one specific message, and none is a bare module tag such as ``[param]``.
    """
    assert [e for e in WARN_ALLOWLIST if re.fullmatch(r"\s*\[\w+\]\s*", e)] == []


def test_parameter_module_fault_trips_the_gate(tmp_path):
    """A parameter-module fault trips the gate: when a PX4 log carries a ``[param]`` storage,
    validation or save failure at warn or error level, the gate counts that line and the example
    fails.
    """
    # A corrupt parameter file, a rejected value and a failed save, as PX4's `param` command prints them.
    log = (
        "ERROR [param] parameter storage is corrupt (-1)\n"
        "ERROR [param] Parameter MPC_THR_HOVER is read-only.\n"
        "ERROR [param] Param save failed (-1)\n"
    )
    assert px4_warnings(_write(tmp_path, log)) == [
        "ERROR [param] parameter storage is corrupt (-1)",
        "ERROR [param] Parameter MPC_THR_HOVER is read-only.",
        "ERROR [param] Param save failed (-1)",
    ]


def test_benign_absent_parameter_line_stays_exempt(tmp_path):
    """The benign missing-parameter line stays exempt: when a PX4 log carries the airframe's line
    for a parameter the build lacks, the gate skips that line.
    """
    assert px4_warnings(_write(tmp_path, f"{ABSENT_PARAM}\n")) == []


def test_every_other_sim_warning_still_gates(tmp_path):
    """Every other simulation warning still gates: when a PX4 log carries a pre-arm warning outside
    the allowlist, mag interference for one, the gate counts the line, and the ``px4_warnings``
    stat means the same as today.
    """
    log = "WARN  [health] Preflight Fail: Strong magnetic interference\n"
    assert px4_warnings(_write(tmp_path, log)) == [
        "WARN  [health] Preflight Fail: Strong magnetic interference",
    ]


def test_ekf2_missing_data_still_gates(tmp_path):
    """An ``ekf2 missing data`` line still gates wherever it fires: when a PX4 log carries the line,
    the gate counts it, since EKF2 missing its data is a real sim fault.
    """
    log = "WARN  [health_and_arming_checks] Preflight Fail: ekf2 missing data\n"
    assert px4_warnings(_write(tmp_path, log)) == [
        "WARN  [health_and_arming_checks] Preflight Fail: ekf2 missing data",
    ]
