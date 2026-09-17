"""The Operator plane, Plane 5: *who commands the vehicle*. A peer to controllers/, who *flies*
it. The ``Operator`` protocol is synchronous/transport-agnostic; ``InProcessOperator`` commands an
in-process autopilot through ``controller.accept_setpoint``, ``Px4Offboard`` commands PX4 over MAVLink
on port ``:14540``. This replaces the old ``_src/pilots/``.
"""

from .in_process import InProcessOperator
from .operator import BaseOperator, Operator, wait_until

__all__ = ["BaseOperator", "InProcessOperator", "Operator", "Px4Offboard", "wait_until"]


def __getattr__(name: str):
    # Px4Offboard pulls `pymavlink`, a dependency for deployment and Software In The Loop (SITL) only,
    # so the in-process operator path, policy, pid or mpc, must not eagerly require it. Import it only
    # on first access.
    if name == "Px4Offboard":
        from .px4_offboard import Px4Offboard

        return Px4Offboard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
