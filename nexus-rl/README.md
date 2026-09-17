# `nexus-rl`, Astro Max reinforcement learning with Isaac Lab on standalone Newton

Trains a **multi-rotor RL hover policy** for the Freefly **Astro Max** on **Isaac Lab over the standalone
Newton physics backend**. It runs without Kit and headless, with *no Isaac Sim and no PhysX*, and records
the training as a Rerun `.rrd` showing the swarm of agents learning. The exported policy, `policy.pt` or
`policy.onnx`, is what the `TrainedPolicyController` example in
`nexus/examples/controllers/policy/controller.py` loads. Deploy it with
[`nexus/examples/controllers/policy/goto/flight.py`](../nexus/examples/controllers/policy/goto/flight.py).

This is a **separate `uv` project** with its own virtual environment, so Isaac Lab, a heavy,
prerelease-pinned stack, stays **out of the core `nexus` environment**. It depends on `nexus`
via an editable path source, so training and the core's deploy share the same Newton 1.3.0, which gives
FR-7 parity. It follows the Isaac Lab external-project layout: a registered task package under
`src/nexus_rl/tasks/direct/<task>/`, holding the env, the config, and `agents/`, and task-agnostic
runner scripts under `scripts/`.

There is no shipped Isaac-Lab-on-Newton drone task to copy, since Newton's task set is cart-pole, ant,
and locomotion. The canonical path is to take the stock `Isaac-Quadcopter-Direct-v0` direct-workflow task
and run it on Newton. This is that, made to actually learn.

## Layout

```
nexus-rl/
├── pyproject.toml                         # nexus-rl: deps nexus (path) + isaaclab[all,newton]
├── config/extension.toml                  # Isaac Lab extension metadata
├── src/nexus_rl/tasks/direct/goto/    # the GoTo task: fly to a sampled goal + hold (vehicle-agnostic)
│   ├── quadcopter_newton.py               # reusable Newton-backend base (kitless; the 5 adaptations below)
│   ├── goto_env.py                        # GoToEnv/Cfg: the single-body per-rotor model (no vehicle)
│   ├── agents/rsl_rl_ppo_cfg.py           # GoToPPORunnerCfg (entropy_coef reliability fix)
│   └── config/astro_max.py                # the Astro Max vehicle config (robot USD + FRD frame) + gym.register
│                                          #   ("Newton-AstroMax-GoTo-Direct-v0"); a new vehicle = a new config
└── scripts/
    ├── rsl_rl/train.py                     # kitless PPO trainer; exports policy.pt + policy.onnx
    ├── rsl_rl/record_demo.py              # records the swarm-improving Rerun .rrd (the demo below)
    └── probe_hover.py                     # open-loop dynamics probe (sanity-check the single-body model)
```

## Environment

Everything runs on the host via the project's own virtual environment: no container, no Isaac Sim
install. `uv run --project nexus-rl <cmd>` resolves and installs Isaac Lab on the Newton backend,
plus Newton, Warp, rsl_rl, and torch, on first use, then runs `<cmd>` in that environment. It needs a
CUDA GPU, and runs on Linux-x86_64 only.

> Isaac Lab `3.0.0b2` *declares* older Newton and Warp pins, but its code runs on the framework's newer
> versions. So this project's `[tool.uv]` override co-resolves the whole stack into **one** Newton, and
> training and the standalone deploy then share the same Newton, for better FR-7 parity.

The Astro Max Universal Scene Description (USD) file is a content-addressed registry asset. It
**auto-resolves** on the host, since the framework's pydantic and registry dependencies are present,
caching under `assets/cache/` in the repo, which `$NEXUS_ASSET_CACHE` overrides. Pass
`--vehicle_usd /path/to/astro_max.usdz` to train.py only to override the lookup.

## Train

```bash
uv run --project nexus-rl python nexus-rl/scripts/rsl_rl/train.py \
  --num_envs 2048 --max_iterations 150 --log_dir .rl-artifacts/rl
# -> kitless (needs_kit=false), physics=NewtonCfg; writes .rl-artifacts/rl/exported/policy.{pt,onnx}
```

**Result:** trains to a clean hover, with `success_rate → 1.0`, reward ~129, and full 500-step
episodes, by about iteration 100, **reliably across seeds**, matching the stock task's native-PhysX
reference. Reliability comes from a small entropy bonus, `GoToPPORunnerCfg.entropy_coef`, plus
start-state randomization, the config field `init_randomize`. See [How it works](#how-it-works) #5.

## Record the demo `.rrd`

```bash
uv run --project nexus-rl python nexus-rl/scripts/rsl_rl/record_demo.py \
  --log_dir .rl-artifacts/rl --out astromax_rl.rrd
```

This loads the checkpoints `train.py` saved and replays the policy at a set of iterations. It logs the
**real Astro Max meshes** for a swarm of agents to a Rerun recording via Newton's `ViewerRerun`, the
same viewer the framework's `RerunLogger` drives. Scrub `time` to watch the swarm fly. The hover-success
and mean-distance curves rise over `train_iter`. Upload it next to the other example recordings:

```bash
aws s3 cp astromax_rl.rrd "s3://$NEXUS_BUCKET/public/ci/logs/astromax_rl.rrd" \
  --content-type application/octet-stream --cache-control max-age=300
# (what scripts/ci/run_rl_example.sh does with UPLOAD=1)
```

## How it works

The stock `Isaac-Quadcopter-Direct-v0` is PhysX-only and doesn't learn on Newton out of the box.
`QuadcopterNewtonEnv` adds, vehicle-agnostically:

1. **Newton MJWarp physics and hygiene without Kit**: sets the `physics=NewtonCfg` preset the stock
   config lacks, and disables every `omni.kit` path, `debug_vis`, `ui_window_class_type`, fabric cloning,
   and visualizers, so `launch_simulation()` never boots Isaac Sim.
2. **Collective Thrust and Body Rates (CTBR) action**: `action[0]` is collective thrust, and
   `action[1:4]` are the commanded body rates, which an inner proportional rate loop tracks with
   `τ = I · gain · (ω_des − ω)`. Newton has no angular-velocity clamp, so raw-torque actuation spins to
   `NaN` and won't learn. The rate loop's error term *is* the damping. The body inertia, read from the
   model, scales the loop gain, so it transfers unchanged across airframes. The params on the config
   are `omega_max`, `rate_gain`, and `gyro_ff`.
3. **World-frame wrench**: because Newton's wrench composer applies forces in the **world** frame, a
   body-z thrust would point straight up regardless of tilt. The drone could change altitude but never
   translate to the goal. The env rotates the body-frame thrust and torque to world before applying
   them. *This was the single bug that blocked convergence.*
4. **`NaN` robustness**: a finite-state termination and reward sanitize, so the inevitable early-
   training crashes can't poison rsl_rl's `check_nan` and end the run.
5. **Reliable convergence**: a small entropy bonus, `GoToPPORunnerCfg.entropy_coef`, which the stock
   config ships as `0`, plus **start-state randomization** in `_reset_idx`, since the stock reset starts
   every env at the *identical* default pose. Without both, a bad seed settles into hovering in place
   and stops improving: full-length episodes, distance stuck at ~1.4 m, `success ≈ 0`. With them,
   `success → 1.0` reliably across seeds. Tunable via the config `init_*` fields and `--entropy`.

The vehicle config, `config/astro_max.py`, supplies the Astro-Max-specific bits: the robot USD, the
Forward Right Down (FRD) frame with `base_body="body_frd"` and `thrust_sign=-1`, and a custom spawner
`func`. That `func` disables the world→base `PhysicsFixedJoint` the registry USD bakes in, which would
otherwise pin the base, so thrust moves nothing. The thrust map, `ct`, `cd`, and `rpm_max`, comes from
the `freefly:actuator:*` attributes of the USD, the same source the core uses, no duplication. A
different vehicle is just another `config/<vehicle>.py`.

## Deploy, the FR-7 round trip

The `TrainedPolicyController` example loads the exported `policy.pt`, and
[`nexus/examples/controllers/policy/goto/flight.py`](../nexus/examples/controllers/policy/goto/flight.py)
flies it in the **standalone** runtime. That closes the loop: train on Isaac-Lab-on-Newton → deploy on
the core. Two shared "single source of truth" pieces make it transfer with no convention change:

- **Observation**: the same `observation_from_state`, in `nexus.examples._lib.observation`, builds
  the 12-D kinematic obs plus the last action, and training and deploy both call it.
- **Action**: the policy's action is **CTBR**, collective thrust plus commanded body rates, and the
  deploy actuator applies it. That actuator mirrors the `_pre_physics_step` of the training env: thrust
  along ±body-z and an inner P rate loop `τ = I·gain·(ω_des − ω)`.

```bash
uv run --extra policy python nexus/examples/controllers/policy/goto/flight.py \
  --policy .rl-artifacts/rl/exported/policy.pt   # the waypoint tour is WAYPOINTS in the script
# -> ~/.cache/nexus/logs/nexus-<timestamp>.rrd  (open with `rerun <path>`)
```
