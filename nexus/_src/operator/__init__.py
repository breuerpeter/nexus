"""The wall-clock wait a script uses on a link's telemetry, ``wait_until``. No operator type lives
here. A controller that takes setpoints flies its guidance, :mod:`nexus._src.guidance`, in the loop.
A script commands PX4 with its own client, which lives in the PX4 peer's folder, ``peers/px4_sitl/``.
"""

from .operator import wait_until

__all__ = ["wait_until"]
