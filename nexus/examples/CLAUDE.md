Every example here is a self-contained, **zero-arg** script: its vehicle, scene, and waypoints live
in the script, and it records its flight `.rrd`. The launcher map is `_EXAMPLES` in `__init__.py`.

- A non-PX4 controller example, `controllers/<name>/`, self-assembles its orchestrator in its own
  `assembly.py` and enters via `Sim.from_orchestrator`.
- Every component states its per-tick work as `stages()`, a list of `Stage` from
  `nexus._src.core.interfaces`. A controller that solves on the host returns `peer_stages(self)`
  from `nexus._src.core.stages`: the Model Predictive Control (MPC), acados, and policy examples.
  Those are a `bind` device stage that binds `tick.controls` to a persistent `(1, 16)` command
  buffer, and the `read` and `exchange` host stages over its `exchange(meas, t, timeout)`. A
  device-native law, the Proportional Integral Derivative (PID) example, states a device stage
  that sets `tick.controls` to its persistent `(1, nr)` command buffer. The loop never calls
  `exchange` itself. A controller with a peer, one with an `attached` flag, connects after the
  capture, so its device buffers exist from construction. One without a peer connects before the
  warm pass.
- `_lib/` is the shared machinery. It holds the single-body `Rotors` actuator in `rotors.py` plus
  the mixers in `mixer.py`, and the Universal Scene Description (USD) to single-body collapse in
  `single_body.py`. It also holds the reference planners in `reference.py` and `min_snap.py`, the
  train↔deploy observation source in `observation.py`, which `nexus-rl` also imports, and the
  evaluation dump in `eval_dump.py`.
- The PX4 examples, `controllers/px4/flight.py`, `flight_manual.py`, and `mission.py`, are plain
  scripts against `nexus.px4.Px4Offboard`. Each opens the client itself on the address in
  `sim.ports["offboard"]`, waits for PX4's heartbeat with `sim.wait_until`, and closes it at the
  end. There is no separate flight driver. CI flies `flight.py` as `px4_sitl`.
- **Examples CI**: `scripts/ci/evaluate_examples.py` runs each example in its `DEFAULT_SET`,
  scores it on `evo` pose Absolute Pose Error (APE), the example's stats, and the steady
  Real Time Factor (RTF), and gates against `scripts/ci/examples_baselines.json`. An example dumps
  its trajectory and stats with `_lib/eval_dump.py`. Only the harness imports `evo`, which is
  GPLv3 and lives in the `ci` dependency-group. The `gpu-examples` workflow flies one example per
  GPU box, `--only <name>`, and `scripts/ci/merge_eval_parts.py` joins the boxes' dumps into the
  one `examples-eval` artifact.
