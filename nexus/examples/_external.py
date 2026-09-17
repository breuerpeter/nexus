"""Where this machine's external tools live: the one definition of the defaults.

These are machine-description facts, so they stay env vars: ``ACADOS_SOURCE_DIR``, acados' own
convention for its install. Every Python consumer, the acados example and
``scripts/ci/evaluate_examples.py``, reads this accessor instead of re-declaring a default that
drifts. Import-light by design, stdlib only: the CI harness calls it pre-flight, before any
runtime imports.

The PX4 checkout and image, ``PX4_DIR`` and ``PX4_IMAGE``, follow the same rule but live with the
launcher that mounts them, :mod:`nexus._src.vehicle.controllers.px4.sitl`.
"""

from __future__ import annotations

import os
from pathlib import Path


def acados_dir() -> Path:
    """The acados install: ``$ACADOS_SOURCE_DIR``, acados' own env convention. The default is where
    ``scripts/setup_acados.sh`` installs to.
    """
    return Path(os.environ.get("ACADOS_SOURCE_DIR") or Path.home() / ".cache" / "nexus" / "acados")
