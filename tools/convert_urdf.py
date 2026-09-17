"""Convert a Unified Robot Description Format (URDF) file to Universal Scene Description (USD) via
Isaac Lab's UrdfConverter. Run once; the USD is a generated artifact Isaac Lab spawns from. The Astro
Max RL example, ``nexus`` and ``examples/isaac-lab``, spawns from this USD, which ships in the
asset registry, so this is a **rare offline regeneration step**, not part of the train/deploy flow.
It needs the **full Isaac Sim**, because Isaac Lab's URDF importer is Kit/GUI-based, so the kitless
Newton install can't run it, which is why it lives here rather than in nexus's host-side ``uv``
example. See README.md for how to run it.

Usage, inside an Isaac Lab image that bundles Isaac Sim, for example nvcr.io/nvidia/isaac-lab:3.0.0-beta2:
    URDF=/path/to/astro_max.urdf OUT=/work/out \
      /isaac-sim/python.sh tools/convert_urdf.py
"""

import os

from isaaclab.app import AppLauncher

_app = AppLauncher(headless=True).app  # the URDF importer runs under a headless Kit app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402


def main():
    urdf = os.environ.get("URDF", "/path/to/astro_max.urdf")
    out = os.environ.get("OUT", "/work/out")
    name = os.environ.get("USD_NAME", os.path.splitext(os.path.basename(urdf))[0] + ".usda")
    cfg = UrdfConverterCfg(
        asset_path=urdf,
        usd_dir=out,
        usd_file_name=name,
        fix_base=False,
        merge_fixed_joints=False,
        make_instanceable=False,
    )
    conv = UrdfConverter(cfg)
    print(f"[convert_urdf] wrote {conv.usd_path} (exists={os.path.exists(conv.usd_path)})")
    _app.close()


if __name__ == "__main__":
    main()
