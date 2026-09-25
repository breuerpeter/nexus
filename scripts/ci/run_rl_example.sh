#!/usr/bin/env bash
# Run the full Astro Max RL example end-to-end and emit the recordings + regression stats.
# Runner-agnostic (like evaluate_examples.py): needs a CUDA host with `uv` — no Docker, no Isaac Sim. Train +
# record run **kitless on the host** in the separate `nexus-rl` uv project (Isaac Lab on the Newton
# backend, its own venv, resolved on demand by `uv run --project nexus-rl`); the fresh export deploys
# on standalone Newton in the core venv, concurrently with the recording. The gpu-rl workflow calls
# this via gpu-runner.yml; it runs check_rl_stats.py at the end.
#
# Env (all optional): REPO (repo root), VEHICLE_USD (auto-resolved if unset), OUT (artifact dir),
#   ENVS, ITERS, SEED, UPLOAD=1 (S3-upload .rrds).
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
OUT="${OUT:-$REPO/.rl-artifacts}"
# 300 iters, not 150: at 150 the hover success is still climbing (~0.88) and the exported policy is
# right at the deploy tour's robustness edge — run-to-run training nondeterminism flipped the leg
# between 3/3 waypoints (err 0.013 m) and 1/3 (err 3.1 m). The extra ~67 s of training buys margin.
ENVS="${ENVS:-2048}"; ITERS="${ITERS:-300}"; SEED="${SEED:-42}"
mkdir -p "$OUT"

# Resolve the Astro Max USD into the asset cache if not supplied (so the GPU caller is a one-liner);
# passed to the RL scripts as --vehicle_usd (VEHICLE_USD pre-seeds it, e.g. a local unpublished asset).
if [ -z "${VEHICLE_USD:-}" ]; then
  VEHICLE_USD="$(cd "$REPO" && uv run --extra policy python -c \
    "from nexus._src.config import LaunchConfig, resolve; print(resolve(LaunchConfig().set_vehicle('astro_max_base')).vehicle_usd_path)")"
fi

echo "===== TRAIN + RECORD SWARM (host, kitless, nexus-rl project) ====="
# `uv run --project nexus-rl` resolves + installs Isaac Lab on the Newton backend on demand (own
# venv, no container), then runs. The trained checkpoints land under --log_dir; record_demo replays them.
( cd "$REPO" && uv run --project nexus-rl python nexus-rl/scripts/rsl_rl/train.py \
    --num_envs "$ENVS" --max_iterations "$ITERS" --seed "$SEED" --save_interval 10 \
    --vehicle_usd "$VEHICLE_USD" \
    --log_dir "$OUT/rl" --stats_json "$OUT/train_stats.json" )

echo "===== RECORD THE SWARM + DEPLOY THE FRESH POLICY (concurrent, host) ====="
# Both read only the checkpoints training wrote and neither waits on the other, so they share the
# box. The swarm recording replays the checkpoints on Isaac Lab. The examples evaluation harness
# flies the fresh export on standalone Newton as goto_policy_fresh, recording the .rrd and scoring
# it (pose APE, tracking error, RTF, control-loop throughput) with every number reported and none
# gated: the flight is one draw from a training that is not reproducible, so it fails only when the
# export does not fly. The hosted policy's flight, the exact deploy gate, rides the gpu-examples
# leg. --upload publishes the .rrd and the bench entries.
( cd "$REPO" && uv run --project nexus-rl python nexus-rl/scripts/rsl_rl/record_demo.py \
    --log_dir "$OUT/rl" --out "$OUT/astromax_rl.rrd" --vehicle_usd "$VEHICLE_USD" ) &
record=$!
uv run --group ci --extra policy \
  python "$REPO/scripts/ci/evaluate_examples.py" --only goto_policy_fresh --out "$OUT" \
  --policy "$OUT/rl/exported/policy.pt" \
  $([ "${UPLOAD:-0}" = 1 ] && echo --upload) &
deploy=$!
record_rc=0; deploy_rc=0
wait "$record" || record_rc=$?
wait "$deploy" || deploy_rc=$?
if [ "$record_rc" != 0 ] || [ "$deploy_rc" != 0 ]; then
  echo "record_demo exit $record_rc, goto_policy_fresh exit $deploy_rc"
  exit 1
fi

echo "===== REGRESSION CHECK (train) ====="
# The learning metrics gate with a margin; the wall-clock stats print and never gate.
python3 "$REPO/scripts/ci/check_rl_stats.py" --train-stats "$OUT/train_stats.json"

if [ "${UPLOAD:-0}" = 1 ]; then
  echo "===== UPLOAD (training swarm recording; via the runner's instance profile) ====="
  : "${NEXUS_BUCKET:?UPLOAD=1 needs NEXUS_BUCKET: the artifacts bucket to write to}"
  aws s3 cp "$OUT/astromax_rl.rrd" "s3://$NEXUS_BUCKET/public/ci/logs/astromax_rl.rrd" \
    --content-type application/octet-stream --cache-control max-age=300
fi

echo "artifacts in $OUT: astromax_rl.rrd, train_stats.json, goto_policy_fresh/goto_policy.{json,npz}, goto_policy_fresh.rrd, benchmark.json"
