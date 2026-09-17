"""Plain stdlib logger for the framework: headless-safe, no Rerun dependency.

The bridge's logger logged through a Rerun handler; observability is a
cross-cutting concern owned by the Recorder/Renderer, so core keeps a plain
stderr logger and the render-null package provides the null log sink.
"""

import logging

logger = logging.getLogger("nexus")  # internal handle; the formatter below sets the "sim/" display namespace
if not logger.handlers:
    _h = logging.StreamHandler()
    # "[sim/<module>]" matches the rerun tee's rows and PX4's "[px4/...]" namespacing. The
    # logger name "nexus" is an internal handle, not display branding.
    _h.setFormatter(logging.Formatter("%(asctime)s [sim/%(module)s] %(levelname)s %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)
