"""Shared example machinery: code that more than one example needs but the core doesn't ship.

Examples import from here with absolute imports, ``from nexus_uav.examples._lib import ...``,
which work under every launcher: ``python -m nexus_uav.examples`` and plain
``python path/to/flight.py``. Ships in the wheel as the examples themselves do.

:mod:`articulated` holds the articulated plant as a differentiable rollout, which the gradient examples fly;
:mod:`mixer` holds ``RotorMixer``, the Collective Thrust and Body Rate (CTBR) and moment mixers,
and the allocation builders. A planner that models the vehicle as one rigid body keeps that model here:
:mod:`motor` holds the first-order Ω lag and its time constant from the vehicle's motors, :mod:`coupling`
the single-body wrench kernel, and :mod:`single_body` the collapse from Universal Scene Description (USD)
to a single body. The reference planners :mod:`reference` and
:mod:`min_snap`, the shared observation source :mod:`observation`, the evaluation dump
:mod:`eval_dump`, and :mod:`origin`, the override layer that moves a shipped scene's geodetic origin,
live here too.
"""

from nexus_uav.examples._lib.coupling import rigid_body_wrench_world  # noqa: F401
from nexus_uav.examples._lib.eval_dump import dump_run, dump_stats  # noqa: F401
from nexus_uav.examples._lib.mixer import (  # noqa: F401
    Controls,
    CtbrMixer,
    CtbrParams,
    MomentMixer,
    RotorMixer,
    build_allocation,
    build_rotor_mixer_from_layout,
    build_rotor_mixer_from_model,
    ctbr_rate_loop,
    ctbr_to_cmd_batched,
    moment_to_cmd_batched,
    pack_vec4,
    wrench_to_cmd,
)
from nexus_uav.examples._lib.motor import lag_step, motor_alpha  # noqa: F401
