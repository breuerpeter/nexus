"""GoTo task: fly a quadrotor to a sampled goal and hold it, in the Direct workflow on the Newton backend.
The env, ``goto_env.GoToEnv``, is vehicle-agnostic; importing this package registers each vehicle variant.
"""

from .config import astro_max  # noqa: F401  registers Newton-AstroMax-GoTo-Direct-v0
from .goto_env import GoToEnv, GoToEnvCfg

__all__ = ["GoToEnv", "GoToEnvCfg"]
