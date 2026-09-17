"""Author variant 4, the Flux payload: astro_max_fpv + an RTX OmniLidar at the LR1 pose, Kit-only.

The experimental ``isaacsim.sensors.experimental.rtx.Lidar.create`` authors a schema-complete
OmniLidar, the ``OmniSensorGenericLidarCoreAPI`` schema + a scan-profile config, and also bakes
the load-bearing ``tick_rate == scanRateBaseHz`` pairing. pxr alone can't replicate the profile
attrs, so this step runs in the isaacsim container against the locally authored variant 2.

Run after author_astro_variants.py:
  uv run nexus script scripts/assets/author_lidar_variant.py
"""

import pathlib
import sys

import omni.kit.commands
import omni.usd
from pxr import Gf, UsdGeom

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
# The repo-root imports and the Kit extension modules below load where they become
# available, so E402 is the price of a top-to-bottom script.
from scripts.assets.author_astro_variants import LR1_OFFSET as DOWN_OFFSET  # noqa: E402
from scripts.assets.convert import _find_base_body  # noqa: E402

src = REPO / "assets" / "local" / "astro_max_fpv.usdz"
out = REPO / "assets" / "local" / "astro_max_fpv_flux.usdz"
assert src.exists(), f"{src} missing: run author_astro_variants.py first"

ctx = omni.usd.get_context()
ctx.open_stage(str(src))
stage = ctx.get_stage()
body = _find_base_body(stage)
lidar_path = body.GetPath().AppendChild("Flux")  # payload naming: the Freefly Flux lidar

# The current, 6.x experimental, authoring API: the deprecated IsaacSensorCreateRtxLidar path
# produces a sensor whose core never ticks in 6.0.1. Lidar.create authors the full schema and the
# load-bearing tick_rate == scanRateBaseHz pairing; a mismatch silently breaks scan accumulation.
from isaacsim.sensors.experimental.rtx import Lidar  # noqa: E402

Lidar.create(str(lidar_path), config="Example_Rotary", tick_rate=10.0)
# Same pose as variant 3's Lr1Cam: below the body, +z in the Forward Right Down (FRD) frame,
# aligned with the body x axis. The lidar scans in its own frame; identity rotation keeps its
# axes = the body axes.
xf = UsdGeom.Xformable(stage.GetPrimAtPath(lidar_path))
xf.ClearXformOpOrder()
mtx = Gf.Matrix4d(1.0)
mtx.SetTranslateOnly(Gf.Vec3d(*DOWN_OFFSET))
xf.AddTransformOp().Set(mtx)
# The host-seam sample rate as a sensor:* custom attr, since the vehicle Universal Scene
# Description (USD) is the single authority: one sample per full scan, matching the preceding
# baked scan tick_rate.
from pxr import Sdf as _Sdf  # noqa: E402

stage.GetPrimAtPath(lidar_path).CreateAttribute("sensor:rate_hz", _Sdf.ValueTypeNames.Float, custom=True).Set(10.0)

# layer.Export can't write usdz: flatten -> usdc temp -> package, the same as the host script
import tempfile  # noqa: E402

from pxr import Usd, UsdUtils  # noqa: E402

flat = Usd.Stage.Open(stage.Flatten())
for prim in list(flat.GetPseudoRoot().GetChildren()):  # strip Kit's editor cameras/scopes from the asset
    if prim.GetName().startswith("OmniverseKit") or prim.GetName() in ("Render", "OmniKit_Viewport_LightRig"):
        flat.RemovePrim(prim.GetPath())
with tempfile.TemporaryDirectory() as td:
    tmp = pathlib.Path(td) / (out.stem + ".usdc")
    flat.GetRootLayer().Export(str(tmp))
    if not UsdUtils.CreateNewUsdzPackage(str(tmp), str(out)):
        raise RuntimeError(f"usdz packaging failed for {out}")
print(f"[author] wrote {out}", flush=True)
# verify round-trip: the prim + schema survive the export
check = Usd.Stage.Open(str(out), Usd.Stage.LoadAll)
prim = [p for p in check.TraverseAll() if p.GetTypeName() == "OmniLidar"]
schemas = prim[0].GetAppliedSchemas() if prim else []
print(
    f"[author] round-trip: OmniLidar prims={[str(p.GetPath()) for p in prim]} schemas={list(schemas)[:3]}",
    flush=True,
)
code = 0 if prim and any("OmniSensorGenericLidarCoreAPI" in s for s in schemas) else 3

raise SystemExit(code)
