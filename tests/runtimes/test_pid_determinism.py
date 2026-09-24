"""Complete-determinism gate, NFR-11, with the in-process
Proportional Integral Derivative (PID) controller.

S-1/S-3 established that Newton CPU physics is bit-exact, but real-PX4 *armed* flight isn't
bit-reproducible, because of its multi-threaded work-queue interleaving. The FR-7 built-in PID is the
deterministic in-process controller that closes that gap: flown through the unchanged
``Orchestrator.run()`` over the bit-exact Newton CPU backend, two runs of the same setup produce
**bit-for-bit the same** trajectories: the determinism CI gate that doesn't wait on PX4.

Heavy, since it builds + settles a Newton model twice; skipped if ``newton`` isn't importable.
"""

import numpy as np
import pytest

pytest.importorskip("newton")
pytest.importorskip("warp")

from nexus._src.config import LaunchConfig
from nexus._src.runtimes.assembly import build_scenario
from nexus._src.runtimes.launch import resolve_to_vehicle_builder
from nexus.examples.controllers.pid.assembly import build_pid_orchestrator

pytestmark = pytest.mark.usefixtures("warp_cpu")  # the build's force_cpu sets the device; the scope puts it back


def _run(steps: int):
    cfg = build_scenario()
    cfg["physics"]["force_cpu"] = True
    vb, _ = resolve_to_vehicle_builder(LaunchConfig().set_vehicle("astro_max_base"))
    orch = build_pid_orchestrator(
        cfg,
        goal_w=(0.0, 0.0, 1.5),
        max_steps=steps,
        vehicle_builder=vb,
    )
    # Capture per-tick body_q/body_qd via the post-step observer, with no recorder seam: step() mutates
    # physics.state0 in place, so it's the current state at the same per-tick point logging would see.
    q, qd = [], []

    def _capture(view, t, n):
        st = orch.physics.state0
        q.append(st.body_q.numpy().copy())
        qd.append(st.body_qd.numpy().copy())

    orch.on_tick = _capture
    orch.run()
    return np.array(q), np.array(qd)


def test_pid_loop_is_bit_identical():
    steps = 120
    q1, qd1 = _run(steps)
    q2, qd2 = _run(steps)
    assert q1.shape[0] == steps and qd1.shape[0] == steps
    # bit-for-bit the same (max|Δ| == 0) on the Warp CPU backend: the NFR-11 gate
    assert np.array_equal(q1, q2), f"body_q diverged: max|Δ|={np.abs(q1 - q2).max()}"
    assert np.array_equal(qd1, qd2), f"body_qd diverged: max|Δ|={np.abs(qd1 - qd2).max()}"
