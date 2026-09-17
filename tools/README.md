# `tools/`

Helper scripts for the `nexus` repo that don't belong in the host-side package: dev utilities,
plus steps that need the **full Isaac Sim** and Kit rather than the `uv` workflow without Kit. One
such step is `convert_urdf.py`, which rebuilds the Astro Max Universal Scene Description (USD) file
from its Unified Robot Description Format (URDF) source.

## `convert_urdf.py`: regenerate the Astro Max scene description from the robot description

Converts a URDF to USD via Isaac Lab's `UrdfConverter`. The Astro Max RL example, `nexus-rl`,
spawns from this USD. The asset registry hosts the artifact, so this is a **rare offline
regeneration step** and not part of training or deploy.

It needs the **full Isaac Sim** because Isaac Lab's URDF importer **needs Kit and its GUI**, so the
Newton install without Kit that the RL example uses can't run it. That's why it's a standalone
docker-run script here rather than a host-side `uv` example.

### Run

1. Point `URDF` at the vehicle URDF file.

2. Run inside an Isaac Lab image that bundles Isaac Sim. The public image on NVIDIA's `nvcr.io`
   registry needs no auth:

   ```bash
   docker run --rm --gpus all -v "$PWD":/work -w /work \
     -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
     nvcr.io/nvidia/isaac-lab:3.0.0-beta2 \
     bash -lc 'URDF=/work/astro_max.urdf OUT=/work/out \
       /isaac-sim/python.sh /work/nexus/tools/convert_urdf.py'
   ```

   Adjust the mount and the paths to wherever this repo and the URDF file live.

**Env vars:** `URDF` is the input `.urdf`, `OUT` is the output dir, and `USD_NAME` defaults to
`<urdf-stem>.usda`. Writes the `.usda` to `OUT` and prints its path. The converter uses a **free
base**, `fix_base=False` and `merge_fixed_joints=False`, to match how the RL task frees the airframe.
