"""The PX4 log warnings gate for the ``px4_sitl`` example: scan a Software in the Loop (SITL) log
for lines that point to a *simulation* problem. It's CI policy rather than a launcher capability,
because which lines are benign depends on the airframe and this PX4 build, so it lives with the
example, not with :mod:`nexus._src.vehicle.controllers.px4.sitl`.
"""

from __future__ import annotations

import re
from pathlib import Path

_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
# Known-benign lines a warnings gate must not count, so it flags only warnings about the
# *simulation*: mag interference, health / estimator / sensor / arming failures. Each entry is a
# regular expression for one message, searched in the line.
# - "Parameter <name> not found.": the boot-time line `param set` and `param set-default` print
#   when the none_astro_max airframe names a parameter this PX4 build lacks. Every other `param` failure, such as a
#   corrupt file, a rejected value or a failed save, still gates.
# - "no heading reference": the pre-arm Extended Kalman Filter (EKF) convergence transient on a
#   fresh boot; heading aligns from the first mag samples, and the operator's failure-free settle
#   wait rides it out. Allowlisted by content so every other pre-arm warning, mag interference
#   included, still gates.
# - "Too many subscriptions, failed to add: <topic>": the pinned PX4 tree's default logged-topic
#   set exceeds the logger's 255-subscription cap on SITL, so it skips these four topics. It's a
#   logging-capacity artifact, not a sim problem, and none of the four feeds the gates here. Each
#   entry names its topic, so a different topic overflowing still gates.
WARN_ALLOWLIST: tuple[str, ...] = (
    r"Parameter \w+ not found\.",
    "Preflight Fail: no heading reference",
    "Too many subscriptions, failed to add: collision_constraints",
    "Too many subscriptions, failed to add: obstacle_distance",
    "Too many subscriptions, failed to add: obstacle_distance_fused",
    "Too many subscriptions, failed to add: vehicle_mocap_odometry",
)


def px4_warnings(log_path: str | Path) -> list[str]:
    """PX4 log lines at warn or error level, over the whole log, that point to a *sim* problem,
    excluding only the known-benign lines that match a message pattern in ``WARN_ALLOWLIST``.

    Args:
        log_path: The SITL log to scan. An unreadable path yields no warnings.

    Returns:
        The offending lines, stripped, in the order they appear in the log.
    """
    try:
        text = _ANSI.sub("", Path(log_path).read_text(errors="ignore")).replace("\r", "\n")
    except OSError:
        return []
    hits = []
    for line in text.splitlines():
        if not re.search(r"\b(WARN|ERROR)\s+\[", line):
            continue
        if any(re.search(a, line) for a in WARN_ALLOWLIST):
            continue
        hits.append(line.strip())
    return hits
