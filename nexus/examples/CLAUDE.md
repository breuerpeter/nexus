Every example here is a self-contained, **zero-arg** script: its vehicle, scene, and waypoints live
in the script, and it records its flight `.rrd`. The launcher map is `_EXAMPLES` in `__init__.py`.

- A non-PX4 controller example, `controllers/<name>/`, self-assembles its orchestrator in its own
  `assembly.py` and enters via `Sim.from_orchestrator`.
- `_lib/` is the shared machinery. It holds the single-body `Rotors` actuator in `rotors.py` plus
  the mixers in `mixer.py`, and the Universal Scene Description (USD) to single-body collapse in
  `single_body.py`. It also holds the reference planners in `reference.py` and `min_snap.py`, the
  train↔deploy observation source in `observation.py`, which `nexus-rl` also imports, and the
  evaluation dump in `eval_dump.py`.
- The PX4 examples, `controllers/px4/flight.py`, `flight_manual.py`, and `mission.py`, are plain
  scripts against `sim.operator`. There is no separate flight driver. CI flies `flight.py`
  as `px4_sitl`.
- **Examples CI**: `scripts/ci/evaluate_examples.py` runs each example in its `DEFAULT_SET`,
  scores it on `evo` pose Absolute Pose Error (APE), the example's stats, and the steady
  Real Time Factor (RTF), and gates against `scripts/ci/examples_baselines.json`. An example dumps
  its trajectory and stats with `_lib/eval_dump.py`. Only the harness imports `evo`, which is
  GPLv3 and lives in the `ci` dependency-group.
