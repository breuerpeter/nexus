"""Shared example machinery: code that more than one example needs but the core doesn't ship.

Examples import from here with absolute imports, ``from nexus.examples._lib import ...``,
which work under every launcher: ``python -m nexus.examples``, ``nexus script``,
and plain ``python path/to/flight.py``. Ships in the wheel as the examples themselves do.

The single-body actuator world lives here, and the examples that collapse the model to one rigid
body run it: :mod:`rotors` holds the summed-base-wrench ``Rotors`` and ``RigidBodyRotors``;
:mod:`mixer` holds ``RotorMixer``, the Collective Thrust and Body Rate (CTBR) and moment mixers,
and the allocation builders; :mod:`motor` holds the first-order Ω lag; :mod:`coupling` holds the
wrench kernels the RL rollouts also run; :mod:`single_body` holds the collapse from
Universal Scene Description (USD) to a single body. The reference planners :mod:`reference` and
:mod:`min_snap`, the shared observation source :mod:`observation`, and the evaluation dump
:mod:`eval_dump` live here too.
"""

from nexus.examples._lib.coupling import (  # noqa: F401
    rigid_body_wrench_batched,
    rigid_body_wrench_world,
)
from nexus.examples._lib.eval_dump import dump_run, dump_stats  # noqa: F401
from nexus.examples._lib.mixer import (  # noqa: F401
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
from nexus.examples._lib.motor import lag_step, motor_alpha  # noqa: F401
from nexus.examples._lib.rotors import RigidBodyRotors, Rotors  # noqa: F401
