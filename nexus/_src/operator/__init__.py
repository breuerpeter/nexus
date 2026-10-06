"""The Operator plane, Plane 5: *who commands the vehicle*. A peer to controllers/, who *flies*
it. The ``Operator`` protocol is synchronous/transport-agnostic; ``InProcessOperator`` commands an
in-process autopilot through ``controller.accept_setpoint``. A script's own client commands PX4
over its offboard link, ``nexus.px4.Px4Offboard``, which lives in the PX4 peer's folder,
``peers/px4_sitl/``. This replaces the old ``_src/pilots/``.
"""

from .in_process import InProcessOperator
from .operator import BaseOperator, Operator, wait_until

__all__ = ["BaseOperator", "InProcessOperator", "Operator", "wait_until"]
