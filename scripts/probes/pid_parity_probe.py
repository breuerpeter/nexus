"""Stage-4 gate: Proportional Integral Derivative (PID) cross-runtime bit-parity, the same flight
with the same bits everywhere.

The in-process PID over CPU Newton physics is the determinism authority per NFR-11: one seeded,
fixed-step, wall-clock-free assembly. The PID lives with the
examples now, since PX4 is the one core control path, but its assembly is importable from the
wheel, so the same ``build_pid_orchestrator`` from ``nexus.examples.controllers.pid.assembly``
runs verbatim on the host and inside the Isaac Sim container; this probe checks the two produce
``body_q`` trajectories that match bit for bit on CPU.

Two phases sharing one reference file, repo-relative since the container bind-mounts the repo:

  host:       uv run python scripts/probes/pid_parity_probe.py dump
  container:  uv run nexus script scripts/probes/pid_parity_probe.py compare-in-kit

Bit-parity requires the same newton/warp builds on both sides. The container runs the workspace
pins from uv.lock, since the entrypoint installs them and ``nexus script`` applies them at boot,
so `compare-in-kit` measures trajectories that match bit for bit; any divergence it reports is a
regression.
"""

from __future__ import annotations

import pathlib
import sys

REF = pathlib.Path(__file__).resolve().parent / "_pid_parity_ref.npz"
STEPS = 500


def _run_cpu_trajectory():
    import numpy as np

    from nexus._src.config import LaunchConfig
    from nexus._src.runtimes.launch import resolve_scenario
    from nexus.examples.controllers.pid.assembly import build_pid_orchestrator

    lc = LaunchConfig().set_vehicle("assets/local/astro_max_base.usdz")
    lc.runtime.device = "cpu"  # the bit-exact authority
    builder, _resolved, cfg = resolve_scenario(lc)
    orch = build_pid_orchestrator(cfg, vehicle_builder=builder, max_steps=STEPS)
    q = []
    orch.on_tick = lambda s, t, n: q.append(orch.physics.state0.body_q.numpy().copy())
    orch.run()
    return np.asarray(q)


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "dump"
    if mode == "dump":
        import numpy as np

        q = _run_cpu_trajectory()
        np.savez_compressed(REF, body_q=q)
        print(f"[P] host reference: {q.shape} -> {REF}")
        return 0

    # compare-in-kit: `nexus script` has already booted Kit under the workspace pins.
    import numpy as np

    ref = np.load(REF)["body_q"]
    q = _run_cpu_trajectory()
    assert q.shape == ref.shape, f"shape {q.shape} != ref {ref.shape}"
    d = np.abs(q - ref)
    exact = np.array_equal(q, ref)
    print(f"[P] in-kit vs host: bit-identical={exact} max|d|={d.max():.3e} (pos {d[:, :, :3].max():.3e})")
    print(f"[P] {'PARITY OK' if exact else 'DIVERGED: the two sides no longer run the same newton/warp'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
