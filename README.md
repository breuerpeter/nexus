<img alt="nexus: GPU-accelerated drone simulation" src="https://breuerpeter.github.io/nexus/assets/lockup-banner.svg" width="100%">

<br />
<br />
[![PyPI](https://img.shields.io/pypi/v/nexus.svg)](https://pypi.org/project/nexus/)
[![Docs](https://img.shields.io/badge/docs-breuerpeter.github.io-blue.svg)](https://breuerpeter.github.io/nexus/)

[![lint](https://github.com/breuerpeter/nexus/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/breuerpeter/nexus/actions/workflows/lint.yml)
[![cpu-pytest](https://github.com/breuerpeter/nexus/actions/workflows/cpu-pytest.yml/badge.svg?branch=main)](https://github.com/breuerpeter/nexus/actions/workflows/cpu-pytest.yml)
[![gpu-pytest](https://github.com/breuerpeter/nexus/actions/workflows/gpu-pytest.yml/badge.svg?branch=main)](https://github.com/breuerpeter/nexus/actions/workflows/gpu-pytest.yml)
[![docs](https://github.com/breuerpeter/nexus/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/breuerpeter/nexus/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/breuerpeter/nexus/branch/main/graph/badge.svg)](https://codecov.io/gh/breuerpeter/nexus)

[![Newton](https://img.shields.io/badge/Newton-1.3.0-76b900.svg?logo=nvidia&logoColor=white)](https://github.com/newton-physics/newton)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-6.0.1-76b900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/isaac/sim)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-3.0.0b2-76b900.svg?logo=nvidia&logoColor=white)](https://github.com/isaac-sim/IsaacLab)
[![Rerun](https://img.shields.io/badge/Rerun-0.34.1-1a1a1a.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAJb0lEQVR4Aa1Xa4xdVRX%2Bztn7nHPf986D6TQtdKjQWCgKFSEGSWrAkAZ%2BEMVqjA9ICL9QQX%2BYmJiYGIM%2FTIgkRvxh0OA7KgU1xvhIH2gLKK0QtA%2Bhr5lOOzPtfd97nvv4rX3HiZdaTIwns%2Bfcc87ee631rW89tgNei4uPzeWZ91QG98bcQcPARZYDKQffIbXPDozD3wbgH3I49p3D344jTw5nGf43UHzn8kOew75J%2BCWWYVx4fF9FvHsqiB69ev2jJ53hicfmUpUccvK8ISscI1I5cmPvueJ2novcdUWSFWwcEctHPsmVOyNprtwNRRpj38Pka7dEeRi6PjrwrUEVpK2Sh5u0MeHjjus2Uu0jyhVSbpJBI%2BMqsSjm7yFtSo1rLbfoGMcKl61kVmRtptVWPUARESVI6BwF2l%2FLQpSHAxR6bdQrBQzLRSSu1%2FCy9CnN%2BfcarddgiozCkIqEoMZm9E6EpjkFQ1yiLKwCvwWKEl1rvQBkqG5uFfFECaKo8gIBrGCiHGEjVoCj51GoBCheVUde8G%2FUApebZii4OQITwslFQE5BFOBqa7XDjSzszsjLOUa%2BjcVyvvb43RWX%2FUsZUdm6kzvxHhP6NkpYLjcwM9NH9OdF6KU%2BvGsnGzoPUzgRbeNwNXcIPHsXUPM4tP5NXTVyp6Jtjozc8qFqUgt6TEUdy7jc3kVheocKJFBkctC5iOpMFZ1KHdGGSeiFDgYvn4N%2FcQhtOhHyATcKM3EeHF%2BGhlujr6YaQDdEdjGy8FuOC%2BbiaApamJpGmTj4y11ajNEMQcAIGTM7R%2Fku1LoJ6NfOor4pQjzTgN42C%2B9UF%2BEry1SgFSNrhzC9CJP3PynxxQDIkcwfhdr3DRTe8TEE77xlxOoRqdeu2r7Pw2%2Fs4Jydl35cjZPkwgJ6f%2BO%2BG%2BsIXzqL4Dqy6so6Cu%2BeRXS6DZ11aGEzRHpxAMcrrS4D9PotUFsfQnHbrbjcVdhwHwo37MBbXUFpCwXtRHzmZ9yzgsH%2BM1wTwp%2Bro7R9Bq4V3uK4MBxbqMoV1G669S03v0Q4IU8Hw0vm1W57P5zgDuTMYqrqo7f3DMLD561C5EAM046QNqPLycHwyF8RnXoVWdS1zG7cdT%2FcYnnte7w4j%2BZvvgOnsESCJsiGir7fgsaduxDMzNo5U%2Fd8CEs%2F6UE19gOncnT2zqN8MxFIiUBGJdJO8h%2BFX%2Fjh1%2Bi7r0PNHYcqHYE3NTUmvPmnPejt%2Bwr8TU10Tnew8koP4XIfwcQxtJ%2F5Ii4cPLA2d2bXAxi0roF3ZQ1RO0P%2F8Ao5QAWSboIwci4RvvKDbyL9x0ESgt8Ki8jCOTQ%2B%2FpG179HZs4hf%2By68eoBDT5zBkjuFjGEqO00fvICbdjKSnv8WhnObUZxdZ9esf%2FARHP%2FsR1kQiugvMQzTdowkBHomGBOeZ9TwwF44SRdL5xQ2Mu35V799bM75555G7eb11LSLGz9YYxmIIdVKoihPmE%2FKGpUri1j86few%2BeHP2TW6VGK29eEwPySJho76BmHmoafrY5v3Xz%2BO%2FunzNuGsmEmsW%2BmQNONuKk2uR%2FdQ1UagTQ32n3BxNVu2mbRaOYo1NbYuTZhPUs3aE0D3U%2BZ8p4ieP64AszN6SWDrQBQESFYYOpEZmzO96xP4X640ZvbMWG88H7pLy0MnQMctj03KHIWWbtjCkkpF7LCyxfi%2FXHHmstBJlfWg25rtgVtAT1XGJhlHU4G6FGbEVLBtSigaPTanffQIWgsrkOJr8z9%2F2dbEGblBGhhJz8V0YAubVLZY%2BRgmrLiUOdBUoEMhkUMOeNWxzaVpuOhNoMQKKb%2BbegJ1VRqfkxi0v%2F0YprdWEJ3pElpbO20nJET0Jph0Wg462%2B%2B2USBKiiKRG6BH%2Fw%2FdInRLVZFSq677ps3pgi55kZIj0pZJkzExGOfA1LbrcGbbDoSL%2B3FyuYw%2F1m5DW9WgGA3bo1fxrv4JLBWvxx2fepCV3F1b9%2FKz%2BxAyCoaambDt1WGUpgvGEZCuqMnNQsen1h6itIDs8EkUfvxrbP%2FwzrV513%2F6YRz9%2FgwKZ5%2FFPYPfIWFD4%2BYpQkI8P3kr3vuFz4wJX3hjCSlbgGZpms1KDi3sF1%2F130xCKtWhcrEpIFQFFKit%2BPbg7ueJjodbdt1p53lsr7Y99Em0774LC4f%2Bjog5IaiXsOmaDbjihq1jex55ZR7Pf%2FkJFN0E5%2FxZXDN4XUhYs92L%2BGWMhIR9oMsSDuzffBLRQ0n1qXWKF3bvt758z3071ubXN8zacbnrL3tfw4tP%2FgiaYLxQvQWz4SKmQvYDfVrmcdOEAv79yleHMKqSsG6z6fNNxGbD2IRz4BkiQdfc%2FoHb8N%2BuF3%2F5Al56%2BjkYdlaH6tuxcTiPucEpdGm8FiamkkKJ789%2FexLCYulqWhd7LJUZQy9C2QzhpzFKaX%2FUN8r5gF30nt0H0WQpn5ydtPlCXCnhl%2FKekEMicOV8F61f%2FcIS%2Blh1C%2Bb6J7EpPI0On%2FdcsQPOzkcO5DpLEOTRqO%2Bn2T4RKWYDVLIehfYQZJFFqZgxdbDzVVlsEet6NRvCQjHN%2Fk9zjmafKCSUPSQ%2FiyJLpfU47W%2FA24ZvYDMjo0Vu%2FX76fThWuVYQKMClj0MWo0Iui7nEwiz9PptsJqiIczSFSncrWd4jiaTf15wr7bxhGA84YnbUidxVkdHDyGECy3gg8YjwJkJ%2BVbiAc4VZ%2FIHCT1TnUAtb3Jd%2BTAV4LkipQMBsJ41nxhKsEciRxLpEezwR0DqpN%2BKGqiEyydBC7xIhRYV9IUc%2BhEla1gAhsvwvJ32U8wFOFK%2FCvunbsVicRTXqYTOV0gMnaDHVNEbHDvpfjY5emfHgaELKBGSTrCPnQ2XPgVUzQD%2Fu8iwR0nUjJESQyiV7GKnlVDy3CUnOEs1gCq%2F61%2BFI43q06Pt63MHVvRO4Il6iArp8WJlkh%2BDg0UIRol05RLKt4shWDykp4ZfzncpSGx0pz4sDQu2u5n8hoLHzlc2iCcM05QhJ8p5XQZ9pXJE%2Fs8Pz2DiYx2Tc5Df%2FWZ0HzgNRVjyUmqyRigVUQkgm5z7HSS3BhM1JbsFGwHCUhsLluU%2Bxrjt8Tli4EiariPeYmVO4ImtEIXta4p7romVa3sR02mREhVj2p1oDTz9i%2B7C5LzXnCN%2FjKjP3CowOw0%2FOdtIViYB8NSkoZ6SgT2gLDE8hl4SqzZyrKDjO6BArx3Q5aXuyZjXKFO9UrNX0Jg6HxYkH9nx168l%2FAnmCoXr8L2ceAAAAAElFTkSuQmCC&logoColor=white)](https://rerun.io)
[![Python](https://img.shields.io/pypi/pyversions/nexus.svg)](https://pypi.org/project/nexus/)

A GPU-accelerated drone simulation framework on [NVIDIA Newton](https://github.com/newton-physics/newton)
and Omniverse, for automated Software In The Loop (SITL) testing, operator training, and faster
flight-stack development cycles.

## Features

- **GPU-accelerated rigid-body physics** built on NVIDIA Newton + Warp.
- **PX4 SITL flight** in decoupled lockstep over MAVLink, with a determinism gate.
- **Sensors & actuators**, such as an Inertial Measurement Unit (IMU), propeller wrenches, and
  state views, as a reusable, unit-tested framework. See the framework-versus-example rule in
  [`docs/design/conventions.md`](docs/design/conventions.md).
- **Differentiable examples**: gradient-based design optimization and Model Predictive Control (MPC)
  waypoint tracking, plus reinforcement learning on Isaac Lab.
- **Two runtimes**: a headless standalone runtime for CI and an Isaac-Sim-backed
  runtime for photorealistic rendering.

## Installation

nexus requires **Python 3.12**, **Linux**, and a **CUDA-capable GPU**. It ships
on PyPI:

```bash
pip install nexus-sim
```

Optional extras: `policy` adds PyTorch, `examples` adds jerk-limited trajectories,
`acados` adds Nonlinear Model Predictive Control (NMPC) code generation, and `all` equals
`policy` plus `examples`.

Development uses [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/breuerpeter/nexus.git
cd nexus
uv sync                 # install the framework
uv run nexus run   # run a scenario in the standalone runtime
```

## Quickstart

Run the command-line tool with the registry default vehicle, `astro_max_base`, and controller, `px4-sitl`:

```bash
uv run nexus run
```

Run a self-contained example. Every example records a Rerun `.rrd`, and `--list` names them:

```bash
uv run -m nexus.examples sampling_mpc
```

The decoupled SITL flight workflow, Newton plus PX4 SITL plus QGroundControl, with the
port map, lives in [`docs/guide/running.md`](docs/guide/running.md).

## Documentation

The full docs, with the system and interface documentation, the examples, and the project plan,
are at **<https://breuerpeter.github.io/nexus/>**.

## Contributing

Contributions are welcome. See [`docs/guide/contributing/`](docs/guide/contributing/index.md)
for setup and the conventional-commit / pre-commit workflow, and
[`docs/design/conventions.md`](docs/design/conventions.md) for the framework-versus-example rule.

## License

Licensed under the [Apache License 2.0](LICENSE). © 2026 Peter Breuer
The hosted vehicle and scene assets and the Astro photograph in the banner aren't Apache-licensed. See [docs/license.md](docs/license.md).
