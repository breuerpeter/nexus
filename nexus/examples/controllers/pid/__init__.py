"""Built-in differentiable Proportional Integral Derivative (PID) controller, the built-in simple controller.

The deterministic in-process controller that's both the determinism authority and the
design-optimization controller; its gains are the differentiable design parameters. One
control law, :mod:`law`, two consumers: :class:`PidController`, eager, and the Warp ``pid_law``
kernel used by ``nexus/examples/design_opt/gain_tuning.py``.
"""

from .controller import PidController
from .law import (
    DEFAULT_GAINS,
    GAIN_NAMES,
    NUM_GAINS,
    TILT_LIMIT,
    hover_action,
    pid_action_np,
    pid_law,
)

__all__ = [
    "DEFAULT_GAINS",
    "GAIN_NAMES",
    "NUM_GAINS",
    "TILT_LIMIT",
    "PidController",
    "hover_action",
    "pid_action_np",
    "pid_law",
]
