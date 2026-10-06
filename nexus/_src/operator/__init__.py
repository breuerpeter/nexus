"""The ground side of a PX4 run: ``Px4Offboard`` commands PX4 over MAVLink on port ``:14540``, as a
ground station does, and ``wait_until`` waits on its telemetry. A controller that takes setpoints has
no operator: its guidance, :mod:`nexus._src.guidance`, runs in the loop.
"""

from .operator import wait_until

__all__ = ["Px4Offboard", "wait_until"]


def __getattr__(name: str):
    # Px4Offboard pulls `pymavlink`, a dependency for deployment and Software In The Loop (SITL) only,
    # so a run with a setpoint controller must not eagerly require it. Import it only on first access.
    if name == "Px4Offboard":
        from .px4_offboard import Px4Offboard

        return Px4Offboard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
