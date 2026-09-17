"""rsl_rl Proximal Policy Optimization (PPO) runner cfg for the GoTo task: the ``rsl_rl_cfg_entry_point``
the task registration points at.

Subclasses the stock Newton quadcopter PPO cfg and folds in the **one** reliability change this task
needs: a small positive ``entropy_coef``. The stock cfg ships ``entropy_coef=0``, which on this task lets
a bad seed get stuck hovering in place: full-length episodes that never track the goal.
A small entropy bonus keeps the policy exploring long enough to discover goal-tracking. Paired with the
env's start-state randomization, this is what makes convergence reliable across seeds. Override at
runtime with ``--entropy``; see ``scripts/rsl_rl/train.py``.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab_tasks.direct.quadcopter.agents.rsl_rl_ppo_cfg import QuadcopterPPORunnerCfg


@configclass
class GoToPPORunnerCfg(QuadcopterPPORunnerCfg):
    experiment_name = "nexus_rl_goto"

    def __post_init__(self):  # the stock cfg sets fields as class attrs, so there is no base __post_init__ to call
        self.algorithm.entropy_coef = 0.01  # exploration bonus; the stock 0 lets a bad seed lock into hovering in place
