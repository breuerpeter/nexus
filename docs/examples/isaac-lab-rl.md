---
description: "A swarm of parallel agents learning to hover with Proximal Policy Optimization (PPO) on Isaac Lab and Newton, then deploying the trained policy on the core runtime."
---

# Astro Max reinforcement learning with Isaac Lab on Newton

`nexus-rl/` trains a **swarm of parallel agents to hover**, then **deploys** the trained
policy back onto the framework's own runtime. PPO, from `rsl_rl`, trains the **Astro Max** on
**Isaac Lab over the standalone Newton backend**, without Kit, Isaac Sim, or PhysX. The result is
a clean hover that matches the stock task's native-PhysX reference of `success_rate → 1.0`. It's
the full **train-to-deploy round-trip**: train on Isaac-Lab-on-Newton → deploy on the core.

## Training: the swarm learning

The recording replays the policy at increasing training iterations, logging the **real Astro Max
meshes** for the swarm. Scrub `time` to watch the drones go from crashing to converging, and the
hover-success and mean-distance curves climb over `train_iter`.

<iframe src="https://app.rerun.io/version/0.34.1/?url=https://d2837jz4fvtxko.cloudfront.net/public/ci/logs/astromax_rl.rrd" width="100%" height="600" frameborder="0"></iframe>

*Early takes crash and scatter. Late takes all converge on their goals. A blank viewer means the
recording isn't uploaded yet: run the pipeline below, then `scripts/ci/evaluate_examples.py --upload`.*

## Deploy: flying waypoints on the core runtime

The exported `policy.pt` flies through the **standalone runtime**, with `uv` and no container, on
the single-body `RigidBodyRotors` actuator. That actuator is the **same** Collective Thrust and
Body Rate (CTBR) mixer and per-rotor motor model the policy trained against: byte-shared Warp
kernels rather than a reimplementation. The drone tours a sequence of waypoints. The next sphere
shows up only once the drone reaches the current one. **Gold** marks the active sphere and
**green** a reached one. This needs no policy change: the policy is goal-relative, so each
waypoint is a fresh single-goal problem from wherever the drone is.

<iframe src="https://app.rerun.io/version/0.34.1/?url=https://d2837jz4fvtxko.cloudfront.net/public/ci/logs/goto_policy.rrd" width="100%" height="600" frameborder="0"></iframe>

*Props render static. The single-body model tracks per-rotor motor speeds, but a spinning-prop viz
on the collapsed body isn't wired yet. The fully articulated spinning-prop deploy is a separate
tracked item.*

CI-measured deploy stats on the pinned runner, see [Benchmarking](../reference/benchmarking.md):

<!-- example-stats: goto_policy -->

## Key stats

Measured on the pinned GPU CI runner, a g5.2xlarge with 1× A10G. That runner refreshes them, and
[CI](#ci-and-regression) says which of them gate and which only report.

| Metric | Value |
|--------|-------|
| Parallel training envs | **2048** |
| Training throughput | **~120,000** env-steps per second |
| Full training run | 300 iterations in **~123 s** |
| Convergence, hover success | **~0.9** from ~iteration 130 on, climbing sharply from ~iteration 100 |
| Reward at convergence | ~122 over full 500-step episodes |
| Exported policy | 16-D obs of 12 kinematic values plus the last action → 4-D CTBR action, Multi-Layer Perceptron (MLP) 64×64 |
| Deploy control loop | **~150 steps per second**, single drone, `mujoco` solver |
| Deploy tracking error | **0.016 m** final waypoint error, 3 of 3 waypoints reached |

## Reproduce these results

Runs entirely on the host with `uv` on a CUDA GPU: **no container, no Isaac Sim**. Training lives
in the separate `nexus-rl` `uv` project, with its own virtual environment.
`uv run --project nexus-rl` installs Isaac Lab on the Newton backend on demand. See the
[README](https://github.com/breuerpeter/nexus/tree/main/nexus-rl). The
Astro Max Universal Scene Description (USD) file auto-resolves. Pass `--vehicle_usd` to train on a
local override.

```bash
# --- 1) TRAIN: 2048 envs, 300 iters, seed 42; exports policy.{pt,onnx} + model_{it}.pt checkpoints ---
uv run --project nexus-rl python nexus-rl/scripts/rsl_rl/train.py \
  --num_envs 2048 --max_iterations 300 --seed 42 --save_interval 10 --log_dir .rl-artifacts/rl

# --- 2) RECORD THE SWARM (reads the checkpoints from step 1) ---
uv run --project nexus-rl python nexus-rl/scripts/rsl_rl/record_demo.py \
  --log_dir .rl-artifacts/rl --out astromax_rl.rrd

# --- 3) DEPLOY: flies the policy on the standalone runtime + records the waypoint tour ---
uv run --extra policy -m nexus.examples goto_policy \
  --policy .rl-artifacts/rl/exported/policy.pt
```

Without `--policy`, `uv run --extra policy -m nexus.examples goto_policy` flies the hosted policy
the example pins: a content-addressed `.pt` fetched from the catalog's `assets.base` into the asset
cache, sha-verified, and offline after the first fetch.

A `--seed` pins the random draws, the initial weights, the start states and the exploration noise,
and not the trained policy. The Newton physics step on the GPU isn't bit-reproducible, and PPO
turns a last-bit difference into a different policy. Two runs at seed 42 end as two policies with
the same statistics: about 0.99 hover success and a mean reward of 124 to 127 over the 2048 envs.
Their weights differ, and so do their waypoint tours. What stays bit-reproducible is the deploy
flight of one exported policy: fly the same `policy.pt` twice and the trajectories match element
for element. The script defaults set the recorder's snapshot iterations and the 3-waypoint deploy
tour.

## CI and regression

Two GPU legs cover the pipeline, each on an **ephemeral, fixed-GPU Amazon Web Services (AWS) EC2
runner**, a g5.2xlarge with 1× A10G. Each gate reads the kind of quantity it fits:

- **The hosted policy's flight is an exact gate.** `gpu-examples` flies `goto_policy`, the hosted
  policy, beside the other examples through `scripts/ci/evaluate_examples.py`. Given a policy, the
  flight is bit-reproducible. Its waypoint count, final tracking error, and pose
  Absolute Pose Error (APE) gate against `scripts/ci/examples_baselines.json` with tight bounds.
  A deploy-side change shows exactly, on the leg that already runs for the deploy-side paths.
- **The training statistics gate with a margin.** `gpu-rl` runs `scripts/ci/run_rl_example.sh`:
  train, then the swarm recording and the fresh export's flight concurrently, then
  `scripts/ci/check_rl_stats.py` against `nexus-rl/stats_baseline.json`. The final hover success
  and mean reward over the 2048 envs are tight statistics where the policy is one draw, so they
  gate with absolute floors under the measured spread.
- **The fresh export's flight gates on completing only.** The flight is one draw, so the harness
  reports its tour, `goto_policy_fresh`, and gates nothing in it. An export the deploy side can't
  load or fly still fails the leg.
- **No gate reads the wall clock.** Training throughput, the deploy control loop's throughput and
  the real-time factor go to the benchmark feed. No throughput baseline holds on a shared runner.

Unit tests on CPU CI, `tests/ci/test_check_rl_stats.py` and `tests/ci/test_evaluate_examples.py`,
cover both gates' logic. Runner setup and the one-time re-baseline on the A10G live in the
project's GPU-CI runbook.

## How the learning works: reinforcement learning and proximal policy optimization at a high level

Unlike every other controller in the examples, nobody programs anything here to fly.
Reinforcement learning turns flying into an optimization problem. The policy starts as random
noise and gets a scalar **reward** for whatever it does. Gradient ascent on that reward is the only
mechanism by which flight emerges. Concretely, at every 20 ms control step each drone receives an
observation $s_t$ and emits an action $a_t$. The observation is 16-D: body-frame velocity, body
rate, gravity direction, goal offset, and its own last action. The action is 4-D CTBR: collective
thrust plus body rates. The drone then receives a reward $r_t$ that pays for being near the goal,
shaped as $1 - \tanh(d/0.8)$ so the gradient doesn't vanish far away, minus small velocity
penalties that discourage thrashing. The quantity to maximize is the expected discounted return
over the 10 s, 500-step episode:

$$
J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[ \sum_{t} \gamma^{\,t}\, r_t \right],
\qquad \gamma = 0.99
$$

**The policy and the critic.** The policy $\pi_\theta(a \mid s)$ is a 64×64 MLP outputting a
Gaussian over actions. It's stochastic on purpose, since the noise *is* the exploration. A second
64×64 MLP, the **critic** $V_\phi(s)$, learns to predict the return from a state. It exists only to
reduce gradient variance. What scales each action's gradient is the **advantage**, the answer to
"was this action better than what the critic expected?" Generalized Advantage Estimation (GAE)
estimates it by blending temporal-difference errors:

$$
\hat A_t = \sum_{l \ge 0} (\gamma \lambda)^l\, \delta_{t+l},
\qquad
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t),
\qquad \lambda = 0.95
$$

**The one idea of PPO.** Vanilla policy gradient must throw its data away after a single gradient
step. The data came from the *old* policy, and a large update invalidates it, often
catastrophically: one bad step can collapse the policy the data collection depends on. PPO makes
reuse safe by clipping the incentive: with the probability ratio
$\rho_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)$, it ascends

$$
L(\theta) = \mathbb{E}_t\!\left[ \min\!\Big( \rho_t \hat A_t,\;
\operatorname{clip}\!\left(\rho_t,\, 1{-}\epsilon,\, 1{+}\epsilon\right) \hat A_t \Big) \right],
\qquad \epsilon = 0.2
$$

Once the policy has moved ~20 % away from the one that collected the data, the clipped $L(\theta)$
stops rewarding it for moving further. So more than one optimization epoch over the same rollout
stays stable.

**The training rhythm** uses `rsl_rl` and the config in
`nexus-rl/src/nexus_rl/tasks/direct/goto/agents/`. Collect a rollout of 24 steps × 2048
parallel envs = **49,152 transitions**, then run 5 epochs × 4 mini-batches of the clipped update,
and repeat. The learning rate of 5e-4 adapts to hold the policy shift near KL ≈ 0.01. 300
iterations ≈ **15 M env-steps in ~2 min** on the A10G. The massive parallelism is the entire reason
training is this cheap, and it runs on Newton's batched GPU physics with all 2048 drones stepping
in lockstep. A small entropy bonus of 0.01 keeps the action distribution from collapsing early.
Without it, an unlucky seed settles into hovering in place and never explores its way to the goal.
That bonus is the one PPO change this task needed over the stock config.

## What it took to learn reliably on Newton

There is no shipped Isaac-Lab-on-Newton drone task. This is the stock `Isaac-Quadcopter-Direct-v0`
`DirectRL` task brought to Newton and made to learn **reliably**. That took a CTBR action with an
inner rate loop, a world-frame wrench fix, a free-flying base, and `NaN`-robust termination. It
also took start-state randomization plus an entropy bonus for seed-robust convergence. Full detail
in the
[README](https://github.com/breuerpeter/nexus/tree/main/nexus-rl).

The shipped **`GoTo`** task trains on the **single-body per-rotor model**.
The CTBR action runs through an inner rate loop and the allocation matrix $B^{-1}$
into four per-rotor motor-speed states. Those states have first-order lag, $\tau \approx 33$ ms,
and saturation. Forward $B$ re-mixes them into one base-body wrench. The fully articulated route, with real rotor
joints, is numerically intractable in the substep-integrated training env. The substep-diluted
rotor wrench turns into an impulsive kick on the low-inertia rotor bodies. The policy can't observe
the lagged motor state, which is why the observation carries the **last action**. The policy infers
the hidden motor state from what it just commanded. The motor lag, mixer,
and thrust map, read from the vehicle USD, are the **same Warp kernels** the deploy actuator runs,
and the observation builder is a single shared implementation. So the policy transfers to the core
runtime with no convention change. Train and deploy meet the same dynamics and observations, bit
for bit.
