"""nexus-rl: Astro Max RL on Isaac Lab over the standalone Newton physics backend, kitless. A consumer
of the nexus sim core; trains the per-rotor policy and exports it for the core's FR-7
``TrainedPolicyController``. Importing this package registers the RL tasks with gymnasium.
"""

from . import tasks  # noqa: F401  triggers gym task registration
