"""StateSensor: a ground-truth oracle sensor that passes the live ``newton.State`` through the
``Measurement``, for privileged state-feedback controllers.

The Hardware In The Loop (HIL) sensors, the Inertial Measurement Unit (IMU), the Global
Positioning System (GPS), the barometer and the magnetometer, give PX4 noisy physical readings. A
state-feedback controller, such as the Model Predictive Control (MPC) example, instead needs the
*full* ground-truth state, ``body_q`` and ``body_qd``, to seed its planning rollout. Rather than a
privileged side-channel, this sensor writes the live state into ``meas.state`` so the controller
reads it through the same neutral ``exchange(meas)`` seam every controller uses, keeping the Orchestrator
loop unchanged. Never serialized to MAVLink.
"""

from __future__ import annotations


class StateSensor:
    """Write the live ``newton.State`` into ``meas.state``: ground-truth ``body_q`` and ``body_qd``."""

    capturable = True  # a reference passthrough: no device work, no host readback
    _state = None

    def sample(self, state, env, t, out) -> None:
        out.state = state

    # Captured split seam, the host-exchange loop: still a pure reference passthrough. The graph
    # replays update the state object's buffers in place, so the reference stashed at capture
    # time *is* the live state at every read.
    def sample_wp(self, state, env, t) -> None:
        self._state = state

    def read(self, out) -> None:
        out.state = self._state


__all__ = ["StateSensor"]
