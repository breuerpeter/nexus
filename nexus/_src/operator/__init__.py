"""The Operator plane, Plane 5: *who commands the vehicle*. A peer to controllers/, who *flies*
it. The ``Operator`` protocol is synchronous/transport-agnostic; ``InProcessOperator`` commands an
in-process autopilot through ``controller.accept_setpoint``. PX4 is commanded over its offboard
link by a script's own client, ``nexus.px4.Px4Offboard``, which lives with the PX4 SITL peer.
This replaces the old ``_src/pilots/``.
"""

from .in_process import InProcessOperator
from .operator import BaseOperator, Operator, wait_until

__all__ = ["BaseOperator", "InProcessOperator", "Operator", "wait_until"]
