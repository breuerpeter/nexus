# Scripts

Every script the repo tracks lives under `scripts/`. None of them ships in the wheel. Run each one from the repo root.

## Docs

| Script | What it does | Run |
|---|---|---|
| `scripts/check_heading_case.py` | Fails on a docs heading that isn't sentence case. The `heading-case` pre-commit hook runs it on each changed page. | `python scripts/check_heading_case.py docs/**/*.md` |
| `scripts/logo/generate_wordmark.py` | Draws the `nexus` word mark in the Michroma font, `docs/assets/logo.svg` and `logo-light.svg`. | `uv run --with fonttools python scripts/logo/generate_wordmark.py` |
| `scripts/logo/generate_subtitle.py` | Draws the tagline under the word mark, `docs/assets/subtitle.svg` and `subtitle-light.svg`. | `uv run --with fonttools python scripts/logo/generate_subtitle.py` |
| `scripts/logo/generate_lockup.py` | Stacks the word mark over the tagline, `docs/assets/lockup.svg` and `lockup-light.svg`, and sets the dark one on the home page photo, `lockup-banner.svg`. Run it after the word mark and the tagline. | `uv run python scripts/logo/generate_lockup.py` |
| `scripts/logo/generate_favicon.py` | Draws the browser-tab icon, `docs/assets/favicon.ico`. | `uv run --with pillow python scripts/logo/generate_favicon.py` |

`scripts/logo/README.md` says how to change the logo. The social card layout, `scripts/logo/nexus-card.yml`, is the folder `mkdocs.yml` names in `cards_layout_dir`.

## Assets

[Converting scenes](../convert-scenes.md) covers `obj_to_usd.py`, `site_scan_splat.py`, `spawn_site.py` and `prepare_asset_upload.py`. The scripts below build the assets the registry ships. Each writes a file that `prepare_asset_upload.py` then hashes for upload, see [Bringing your own assets](../your-own-assets.md).

| Script | What it does | Run |
|---|---|---|
| `scripts/assets/author_powerline_scene.py` | Builds the `powerline` scene from Poly Haven assets. | `uv run python scripts/assets/author_powerline_scene.py /tmp/powerline.usdz` |
| `scripts/assets/author_slalom_scene.py` | Builds the `slalom` scene: three pillars the sampling Model Predictive Control (MPC) example steers around. | `uv run python scripts/assets/author_slalom_scene.py /tmp/slalom.usdz` |
| `scripts/assets/author_cesium_scene.py` | Builds the `cesium` scene: the Cesium globe with no token. It runs in the Kit image, which it starts itself. | `uv run python scripts/assets/author_cesium_scene.py --out assets/local/cesium.usd` |
| `scripts/assets/convert_urdf.py` | Converts the Astro Max Unified Robot Description Format (URDF) file to the Universal Scene Description (USD) file the RL example spawns. It needs the full Isaac Sim, so it runs inside an Isaac Lab image. | See below. |

To run `convert_urdf.py`, set `URDF` to the input file and `OUT` to the output folder. `USD_NAME` sets the filename and defaults to the input's stem with `.usda`. The public Isaac Lab image needs no login:

```bash
docker run --rm --gpus all -v "$PWD":/work -w /work \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  nvcr.io/nvidia/isaac-lab:3.0.0-beta2 \
  bash -lc 'URDF=/work/astro_max.urdf OUT=/work/out \
    /isaac-sim/python.sh /work/scripts/assets/convert_urdf.py'
```

## CI

The workflows under `.github/workflows/` run these. Each also runs by hand with the command shown.

| Script | What it does | Run |
|---|---|---|
| `scripts/ci/evaluate_examples.py` | Flies the examples and gates each against `scripts/ci/examples_baselines.json`, see [Benchmarking](../../reference/benchmarking.md). | `uv run --group ci --extra examples python scripts/ci/evaluate_examples.py --only pid` |
| `scripts/ci/merge_eval_parts.py` | Joins the results the `gpu-examples` boxes upload, one example each, into one folder. | `python3 scripts/ci/merge_eval_parts.py <parts dir> <out dir>` |
| `scripts/ci/benchmark_cell.py` | Flies one cell of the benchmark matrix: one vehicle, scene, and device. | `uv run python scripts/ci/benchmark_cell.py --vehicle astro_max_base --scene empty --device cpu` |
| `scripts/ci/benchmark_matrix.py` | Flies every cell of the benchmark matrix and writes `docs/data/rtf_matrix.json`. In CI, `--list`, `--cell` and `--merge` split the run across boxes. | `uv run python scripts/ci/benchmark_matrix.py --only physics` |
| `scripts/ci/run_rl_example.sh` | Trains, records and gates the Astro Max RL example, see [Astro Max RL](../../examples/isaac-lab-rl.md). | `bash scripts/ci/run_rl_example.sh` |
| `scripts/ci/check_rl_stats.py` | Gates a training run's statistics against `nexus-rl/stats_baseline.json`. | `python scripts/ci/check_rl_stats.py --train-stats train.json` |
| `scripts/ci/px4_image_cache.sh` | Loads the PX4 Software In The Loop (SITL) image from a GPU box's cache before a run, and saves it after. | `bash scripts/ci/px4_image_cache.sh load` |
| `scripts/ci/discover_aws_runner_config.py` | Lists the Amazon Web Services (AWS) zones, images and subnets where a GPU box can start, for the start step to try in order. Every GPU run calls it through `gpu-runner.yml`. By hand, it helps to debug a box that won't start, and it needs AWS credentials. | `AWS_REGION_CANDIDATES="us-west-2 us-east-1" AWS_INSTANCE_TYPE=g5.2xlarge python3 scripts/ci/discover_aws_runner_config.py` |
