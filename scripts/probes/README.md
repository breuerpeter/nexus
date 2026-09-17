# Invariant probes for Isaac Sim

In-container diagnostics that check load-bearing runtime invariants. Re-run after an Isaac Sim
container upgrade or a change to the capture and render seams:

- `isaac_renderer_probe.py` checks the pure-renderer contract in one pass. First, vehicle pose
  writes and epoch pairing. A body-mounted camera must be pixel-rigid against its own vehicle
  across a teleport, and the render must display the write from one step earlier. The probe reports the
  render-used pose via `camera_params`. Second, captured CUDA-graph physics over the framework's
  own `NewtonPhysics` inside Kit is real. The probe requires motion: replays must climb, not just
  "agree" on a grounded state. Third, mean render cost. Fourth, the Kit timeline clock tracks the
  lockstep sim time. If nothing drives it, Kit's playback paces it off the render frame count, and
  authored Universal Scene Description (USD) time-samples resolve against it. Scene animation then
  runs at the render rate and drifts away from sim time without bound, see GH #62.

- `pid_parity_probe.py` is the cross-runtime bit-parity gate, stage 4 of the runtime unification.
  The same `build_pid_orchestrator` CPU flight on the host with `dump` and inside the container
  with `compare-in-kit` must produce `body_q` trajectories that match bit for bit. Measured as a
  bit-for-bit match, with max|d| = 0.0 over 500 steps, on the WORKSPACE-pinned stack, and,
  historically, even across the bundled-versus-workspace skew.

Run a probe, and read its header for the details: `uv run nexus script scripts/probes/<probe>.py`

Related: `tools/analyze_rrd_video.py`, a host-side script for honest recording analysis: real frame
rate, duplicate fraction, and pinhole and body coincidence. It uses per-row `as_py()` extraction,
never `cell.values`.
