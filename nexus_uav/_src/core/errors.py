"""The error a run raises when it has no verdict: the run never started, or it lost a peer.

A light module of its own, so a peer's runner can subclass it without importing the loop and Warp.
"""


class RunError(RuntimeError):
    """The run failed for a reason outside the script's own check: it never started, or it lost a peer
    before it ended.

    A component never connected or never attached, a peer never answered or exited during the setup, or a
    peer stopped answering mid-run. The message names the cause, then each peer's files, such as its console
    log, and the peer's own error lines, so a script's output says why even after the machine goes away. A
    run that ends on ``max_steps``, on ``stop()`` or at its mission's end raises nothing.

    A script separates this error from its own check failing by catching it::

        try:
            sim.wait_until(op.at_target, sim_timeout=60.0)
        except nexus_uav.RunError:
            ...  # the run broke: report an error, not a failed check
    """
