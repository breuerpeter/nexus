r"""Author the ``powerline`` scene Universal Scene Description (USD): an overhead power line on wooden
poles beside a road at sunset, for flying an inspection along it.

A 50 m square of grassy, rocky ground. A two-lane road, its tarmac raised over the ground, runs
along x down a smooth 3 m drop, turns 90 degrees through a 10 m radius curve and runs along y; the
power line follows it on the inside of the turn, with a corner pole at the turn. Tall firs on their
roots, ferns, stumps, fallen logs, dry branches and rocks stand clear of the road, the line and the
square's edge. The inspection takes off from the road beside the line's end pole on the low ground,
flies the whole line round the corner and up the drop to its other end on the high ground.

The poles, plants, rocks, ground textures and sky are Poly Haven's, all CC0:
``modular_electricity_poles``, ``fir_tree_01``, ``pine_roots``, ``fern_02``, ``tree_stump_01``,
``dead_tree_trunk``, ``dry_branches_medium_01``, ``rock_moss_set_02``, ``boulder_01``,
``aerial_grass_rock``, ``asphalt_track`` and ``belfast_sunset_puresky``. This script fetches them
from Poly Haven's API into ``~/.cache/nexus/polyhaven`` and strings the wires between the insulators
itself, since Poly Haven ships no overhead line.

Unlike the other scenes, whose sky is ``scene_root.author_sky``'s plain dome, this one is lit by a
sky-only sunset High Dynamic Range Image (HDRI): the sunset and its clouds are the scene's point. A
"puresky" image holds no captured ground, so it can't stand in for the scene's own.

Everything renders from Poly Haven's meshes, while the physics collides with invisible shapes: a
capsule along each pole, crossarm and wire span, boxes under the drop and the high ground, and boxes
along the road whose tops are the tarmac. The scene adds about a hundred shapes to the Newton model,
not the meshes' millions of triangles. The plants and rocks carry no colliders. The physics' own
plane is the low ground's collider, 6 cm over the grass, level with the tarmac at the start.

    uv run python scripts/assets/author_powerline_scene.py /tmp/powerline.usdz
then prepare it for upload, which prints the sha, the upload key, and the catalog snippet:
    uv run python scripts/assets/prepare_asset_upload.py --scene /tmp/powerline.usdz
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import sys
import urllib.request
from pathlib import Path

CACHE = Path.home() / ".cache" / "nexus" / "polyhaven"
RES = "2k"
PROP_RES = "1k"  # the props stand away from the camera

HALF = 25.0  # the scene is a square of this half-width [m]

# The ground: DROP_M metres higher than the low ground up to DROP_X[0], easing down to it by DROP_X[1]
# along a half cosine, so the road meets both flats without a crease.
DROP_X = (-22.0, -2.0)
DROP_M = 3.0
DROP_COLLIDERS = 8  # boxes along the drop; their chords stay within 3 cm of the curve

# The road's center line: straight along x at y = ROAD_Y, a quarter circle of ROAD_TURN_R, then
# straight along y at x = ROAD_X.
ROAD_Y = -7.5
ROAD_X = 17.5
ROAD_TURN_R = 10.0
ROAD_HALF_WIDTH = 3.5
ROAD_THICK = 0.06  # the tarmac's top over the ground [m]
ROAD_SKIRT = 0.05  # its sides reach this far below the ground, so no gap shows on a slope [m]
ROAD_COLLIDER = 2.0  # the length of each box the tarmac collides through [m]
EDGE_INSET, LINE_WIDTH = 0.3, 0.15  # road markings [m]
DASH, GAP = 3.0, 6.0

FLOOR = ("aerial_grass_rock", 15.0)  # the ground everywhere: grass over rocky soil

# The line: its poles in order, on the inside of the road's turn.
POLES = ((-22.0, 0.0), (-2.0, 0.0), (10.0, 0.0), (10.0, 12.0), (10.0, 24.0))
# Open ground kept free of props: beside the pole at the foot of the drop, and beside the line's
# far end pole on the high ground, where the inspection lands. The catalog ``start`` is on the
# tarmac beside the line's other end pole, which the props keep clear of already. The physics adds
# its own plane at world z = 0, where ``start`` goes, and a vehicle can't go below it, so the start
# stands on the low ground and the inspection climbs. On the tarmac top, the start leaves the
# plane 6 cm over the grass, which nothing lands on.
CLEAR_SPOTS = ((-2.0, -2.0), (-22.0, -2.0))

# preset_02 in the pole asset is the assembled line pole, its axis at asset (-4.5, 0), its crossarms
# along asset x and its insulator strings along asset y.
POLE_PRESET = "preset_02"
POLE_AXIS_X = -4.5
POLE_HEIGHT = 6.0
POLE_RADIUS = 0.07
CROSSARMS = ((1.2, 5.59), (0.7, 4.49))  # (length across the line, height) [m]
# Each insulator string's tip, measured on the mesh: its offset across the line and its height.
# The tip sits INSULATOR_REACH along the line from the pole's axis, on both sides.
INSULATORS = ((-0.45, 5.69), (0.0, 5.69), (0.45, 5.69), (-0.2, 4.59), (0.2, 4.59))
INSULATOR_REACH = 0.415
WIRE_WIDTH = 0.022  # [m]; thinner, a wire 20 m away falls under a pixel and breaks into dashes
WIRE_SAG = 0.35  # the drop at mid-span of a 20 m span [m]
WIRE_POINTS = 21
WIRE_COLLIDERS = 4  # capsules per span

# Plants and rocks, placed largest first: (Poly Haven asset, the one part of its set kept, texture
# resolution, canopy or footprint radius at scale 1, how many, scale range, how far it sinks below
# the ground so its own ground patch doesn't show). The fir is fir_tree_01's middle tree, 14 m tall
# and full to the ground; its thin needles need 2k maps, or a distant tree's blurred alpha drops
# under the cutoff.
PROPS = (
    ("fir_tree_01", "fir_tree_01_b_LOD0", "2k", 2.9, 30, (0.85, 1.05), 0.15),
    ("rock_moss_set_02", None, PROP_RES, 4.2, 3, (0.8, 1.0), 0.15),
    ("dead_tree_trunk", None, PROP_RES, 1.6, 4, (0.9, 1.1), 0.05),
    ("boulder_01", None, PROP_RES, 1.0, 14, (1.2, 2.2), 0.15),
    ("tree_stump_01", None, PROP_RES, 0.8, 5, (0.9, 1.2), 0.05),
    ("dry_branches_medium_01", None, PROP_RES, 0.7, 8, (0.8, 1.2), 0.02),
    ("fern_02", None, PROP_RES, 0.6, 300, (0.8, 1.3), 0.02),
)
# A root flare at the foot of every fir, sized with it: the half of Poly Haven's pair of flares that
# sits at the set's origin, 12 cm tall, so it stands on the ground rather than sinking.
ROOTS = ("pine_roots", "pine_roots_b")
NEEDLE_CUTOFF = 0.12  # opacity below this is a hole in a needle card; low, so thin needles stay
PROP_SEED = 7
CLEAR = 1.5  # between a canopy and the road's edge, the line or the square's edge [m]
OVERLAP = 0.5  # the share of their radii two props can overlap, as in a stand of trees
PROP_TRIES = 20000

SKY = "belfast_sunset_puresky"
SKY_INTENSITY = 400.0
# The sun at the brightest point of the sky image, measured: 2 degrees over the horizon.
SUN_AZIMUTH_DEG = 37.0  # from the image center's direction, +y, towards -x
SUN_ELEVATION_DEG = 4.0
SUN_INTENSITY = 700.0
SUN_COLOR = (1.0, 0.62, 0.38)


def _get(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "nexus-assets"}))


def _fetch(url: str, md5: str, dst: Path) -> Path:
    if not (dst.exists() and hashlib.md5(dst.read_bytes()).hexdigest() == md5):
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = _get(url).read()
        if hashlib.md5(data).hexdigest() != md5:
            raise ValueError(f"md5 mismatch for {url}")
        dst.write_bytes(data)
    return dst


def fetch_model(asset: str, res: str = RES) -> Path:
    """Fetch a Poly Haven model's USD and its textures; return the local ``.usdc``."""
    u = json.load(_get(f"https://api.polyhaven.com/files/{asset}"))["usd"][res]["usd"]
    for rel, inc in u.get("include", {}).items():
        _fetch(inc["url"], inc["md5"], CACHE / asset / rel)
    return _fetch(u["url"], u["md5"], CACHE / asset / u["url"].rsplit("/", 1)[-1])


def fetch_texture(asset: str) -> dict[str, Path]:
    """Fetch a Poly Haven texture's color, normal and roughness maps as JPEG."""
    files = json.load(_get(f"https://api.polyhaven.com/files/{asset}"))
    out = {}
    for key, name in (("Diffuse", "diffuse"), ("nor_gl", "normal"), ("Rough", "roughness")):
        f = files[key][RES]["jpg"]
        out[name] = _fetch(f["url"], f["md5"], CACHE / asset / f["url"].rsplit("/", 1)[-1])
    return out


def fetch_sky(asset: str) -> Path:
    f = json.load(_get(f"https://api.polyhaven.com/files/{asset}"))["hdri"][RES]["hdr"]
    return _fetch(f["url"], f["md5"], CACHE / asset / f["url"].rsplit("/", 1)[-1])


def height(x: float, y: float) -> float:
    """The ground's height at ``(x, y)``: high, a half-cosine drop along x, low."""
    x0, x1 = DROP_X
    t = min(max((x - x0) / (x1 - x0), 0.0), 1.0)
    return DROP_M * (1.0 + math.cos(math.pi * t)) / 2.0


def road_centre(step: float = 0.5) -> list[tuple[float, float]]:
    """Points along the road's center line, ``step`` apart, from one edge of the scene to the other."""
    cx, cy = ROAD_X - ROAD_TURN_R, ROAD_Y + ROAD_TURN_R  # the turn's center
    pts = [(x, ROAD_Y) for x in _range(-HALF, cx, step)]
    n = max(2, int(math.pi / 2 * ROAD_TURN_R / step))
    pts += [
        (cx + ROAD_TURN_R * math.sin(a), cy - ROAD_TURN_R * math.cos(a))
        for a in (i * math.pi / 2 / n for i in range(n))
    ]
    pts += [(ROAD_X, y) for y in _range(cy, HALF, step)] + [(ROAD_X, HALF)]
    return pts


def _range(a: float, b: float, step: float) -> list[float]:
    return [a + i * step for i in range(int((b - a) / step))]


def _offset(path, d: float):
    """``path`` moved ``d`` to its left, point by point."""
    out = []
    for i, (x, y) in enumerate(path):
        (x0, y0), (x1, y1) = path[max(i - 1, 0)], path[min(i + 1, len(path) - 1)]
        length = math.hypot(x1 - x0, y1 - y0)
        out.append((x - d * (y1 - y0) / length, y + d * (x1 - x0) / length))
    return out


def _distance_to(path, x: float, y: float) -> float:
    return min(math.hypot(x - px, y - py) for px, py in path)


def _surface(stage, path: str):
    from pxr import UsdShade

    mat = UsdShade.Material.Define(stage, path)
    surf = UsdShade.Shader.Define(stage, f"{path}/Surface")
    surf.CreateIdAttr("UsdPreviewSurface")
    mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
    return mat, surf


def _plain_material(stage, path: str, color, *, metallic: float = 0.0, roughness: float = 0.5):
    from pxr import Sdf

    mat, surf = _surface(stage, path)
    surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    return mat


def _uv(stage, path: str, name: str, reader, scale: float):
    from pxr import Sdf, UsdShade

    xf = UsdShade.Shader.Define(stage, f"{path}/{name}")
    xf.CreateIdAttr("UsdTransform2d")
    xf.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    xf.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set((scale, scale))
    return xf


def _textured_material(stage, path: str, maps: dict[str, Path], tile: float):
    """A UsdPreviewSurface whose maps repeat every ``tile`` metres on a mesh with metre UVs."""
    from pxr import Sdf, UsdShade

    mat, surf = _surface(stage, path)
    reader = UsdShade.Shader.Define(stage, f"{path}/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tiled = _uv(stage, path, "Tile", reader, 1.0 / tile)

    def texture(name: str, file: Path, st, color_space: str):
        tex = UsdShade.Shader.Define(stage, f"{path}/{name}")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(file))
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
        return tex

    for name, file in maps.items():
        if name == "diffuse":
            out = texture(name, file, tiled, "sRGB").CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(out)
        elif name == "normal":
            tex = texture(name, file, tiled, "raw")
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((2.0, 2.0, 2.0, 1.0))
            tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set((-1.0, -1.0, -1.0, 0.0))
            out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            surf.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(out)
        else:
            out = texture(name, file, tiled, "raw").CreateOutput("r", Sdf.ValueTypeNames.Float)
            surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(out)
    return mat


def _mesh(stage, path: str, points, faces, uvs, material) -> None:
    """A quad mesh with per-vertex UVs in metres, bound to ``material``."""
    from pxr import Sdf, UsdGeom, UsdShade

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(uvs)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def _strip(stage, path: str, left, right, lift: float, material) -> None:
    """A mesh between two matching point rows, draped ``lift`` over the ground."""
    points, uvs, faces = [], [], []
    along = 0.0
    for i, (a, b) in enumerate(zip(left, right, strict=True)):
        if i:
            along += math.hypot(a[0] - left[i - 1][0], a[1] - left[i - 1][1])
        width = math.hypot(b[0] - a[0], b[1] - a[1])
        points += [(a[0], a[1], height(*a) + lift), (b[0], b[1], height(*b) + lift)]
        uvs += [(along, 0.0), (along, width)]
        if i:
            faces.append((2 * i - 2, 2 * i - 1, 2 * i + 1, 2 * i))
    _mesh(stage, path, points, faces, uvs, material)


def _slab(stage, path: str, left, right, material) -> None:
    """The tarmac: a strip ``ROAD_THICK`` over the ground between two point rows, with its two
    sides down to ``ROAD_SKIRT`` below the ground, so the road reads as a raised layer.
    """
    top_l = [(a[0], a[1], height(*a) + ROAD_THICK) for a in left]
    top_r = [(b[0], b[1], height(*b) + ROAD_THICK) for b in right]
    foot_l = [(a[0], a[1], height(*a) - ROAD_SKIRT) for a in left]
    foot_r = [(b[0], b[1], height(*b) - ROAD_SKIRT) for b in right]
    points, uvs, faces = [], [], []
    for rows in ((top_l, top_r), (foot_l, top_l), (top_r, foot_r)):
        base, along = len(points), 0.0
        for i, (a, b) in enumerate(zip(*rows, strict=True)):
            if i:
                along += math.dist(a[:2], rows[0][i - 1][:2])
                faces.append((base + 2 * i - 2, base + 2 * i - 1, base + 2 * i + 1, base + 2 * i))
            width = math.dist(a, b)
            points += [a, b]
            uvs += [(along, 0.0), (along, width)]
    _mesh(stage, path, points, faces, uvs, material)


def _ground(stage, path: str, material) -> None:
    """The ground as a 1 m grid following ``height``, UVs in metres."""
    n = int(2 * HALF) + 1
    points = [(-HALF + i, -HALF + j, height(-HALF + i, -HALF + j)) for j in range(n) for i in range(n)]
    uvs = [(p[0], p[1]) for p in points]
    faces = [
        (j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i) for j in range(n - 1) for i in range(n - 1)
    ]
    _mesh(stage, path, points, faces, uvs, material)


def _tube(stage, path: str, pts, material, sides: int = 6) -> None:
    """A wire as a thin closed tube along ``pts``: a mesh reads the same from every side, where a
    flat curve ribbon can turn edge-on to the camera and break up.
    """
    from pxr import Gf, UsdGeom, UsdShade

    r = WIRE_WIDTH / 2.0
    points, faces = [], []
    for i, p in enumerate(pts):
        a, b = Gf.Vec3d(*pts[max(i - 1, 0)]), Gf.Vec3d(*pts[min(i + 1, len(pts) - 1)])
        t = (b - a).GetNormalized()
        u = Gf.Cross(t, Gf.Vec3d(0.0, 0.0, 1.0)).GetNormalized()
        v = Gf.Cross(u, t)
        for k in range(sides):
            ang = 2.0 * math.pi * k / sides
            points.append(Gf.Vec3f(*(Gf.Vec3d(*p) + r * (math.cos(ang) * u + math.sin(ang) * v))))
        if i:
            for k in range(sides):
                k1 = (k + 1) % sides
                faces += [(i - 1) * sides + k, (i - 1) * sides + k1, i * sides + k1, i * sides + k]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreateDoubleSidedAttr(True)  # the winding turns inward on some spans, and Kit culls back faces
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * (len(faces) // 4))
    mesh.CreateFaceVertexIndicesAttr(faces)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def _collider_capsule(stage, path: str, a, b, radius: float) -> None:
    """An invisible capsule from ``a`` to ``b``: a collision shape the renderer never draws."""
    from pxr import Gf, UsdGeom, UsdPhysics

    a, b = Gf.Vec3d(*a), Gf.Vec3d(*b)
    axis = b - a
    cap = UsdGeom.Capsule.Define(stage, path)
    cap.CreateRadiusAttr(radius)
    cap.CreateHeightAttr(max(axis.GetLength() - 2.0 * radius, 0.0))
    cap.CreateAxisAttr("Z")
    xf = UsdGeom.Xformable(cap)
    xf.AddTranslateOp().Set((a + b) / 2.0)
    xf.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), axis.GetNormalized()).GetQuat()))
    cap.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(cap.GetPrim())


def _box(stage, path: str, centre, size, tilt_deg: float = 0.0) -> None:
    """An invisible box collider, turned ``tilt_deg`` about y."""
    from pxr import Gf, UsdGeom, UsdPhysics

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box)
    xf.AddTranslateOp().Set(Gf.Vec3d(*centre))
    xf.AddRotateYOp().Set(tilt_deg)
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    box.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())


def _ground_colliders(stage, root: str) -> None:
    """Boxes whose top faces follow the high ground and the drop, chord by chord; the physics' own
    plane at z = 0 is the low ground.
    """
    x0, x1 = DROP_X
    thick = 1.0
    _box(stage, f"{root}/high_ground", ((-HALF + x0) / 2.0, 0.0, DROP_M - thick / 2.0), (x0 + HALF, 2.0 * HALF, thick))
    for k in range(DROP_COLLIDERS):
        xa = x0 + (x1 - x0) * k / DROP_COLLIDERS
        xb = x0 + (x1 - x0) * (k + 1) / DROP_COLLIDERS
        za, zb = height(xa, 0.0), height(xb, 0.0)
        dx, dz = xb - xa, zb - za
        length = math.hypot(dx, dz)
        normal = (-dz / length, dx / length)  # up and back from the chord, in (x, z)
        centre = ((xa + xb) / 2.0 - normal[0] * thick / 2.0, 0.0, (za + zb) / 2.0 - normal[1] * thick / 2.0)
        # About y, a positive turn tips +x downwards: a falling chord needs a positive one.
        _box(stage, f"{root}/drop_{k}", centre, (length, 2.0 * HALF, thick), math.degrees(math.atan2(-dz, dx)))


def _road_colliders(stage, root: str, centre) -> None:
    """Boxes whose top faces are the tarmac, chord by chord along the road, so a vehicle stands on
    the road surface rather than on the ground under it.
    """
    from pxr import Gf, UsdGeom, UsdPhysics

    thick = 0.3
    step = max(1, round(ROAD_COLLIDER / math.dist(centre[0], centre[1])))
    marks = [*centre[::step], centre[-1]] if (len(centre) - 1) % step else centre[::step]
    for k, (a, b) in enumerate(itertools.pairwise(marks)):
        top_a = Gf.Vec3d(a[0], a[1], height(*a) + ROAD_THICK)
        top_b = Gf.Vec3d(b[0], b[1], height(*b) + ROAD_THICK)
        along = (top_b - top_a).GetNormalized()
        across = Gf.Cross(Gf.Vec3d(0.0, 0.0, 1.0), along).GetNormalized()
        up = Gf.Cross(along, across)
        frame = Gf.Matrix3d(*along, *across, *up)  # rows: the box's x, y and z axes in the world
        centre_pt = (top_a + top_b) / 2.0 - up * (thick / 2.0)
        box = UsdGeom.Cube.Define(stage, f"{root}/road_{k}")
        box.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(box)
        xf.AddTranslateOp().Set(centre_pt)
        xf.AddOrientOp().Set(Gf.Quatf(frame.ExtractRotation().GetQuat()))
        # A little longer than its chord, so neighbouring boxes overlap and leave no gap in a turn.
        xf.AddScaleOp().Set(Gf.Vec3f((top_b - top_a).GetLength() + 0.2, 2.0 * ROAD_HALF_WIDTH, thick))
        box.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())


def _place(
    stage, path: str, asset: Path, x: float, y: float, z: float, turn: float, scale: float, *, instanceable: bool
):
    """Reference a whole asset at ``(x, y, z)``, turned about z and scaled."""
    from pxr import Gf, UsdGeom

    prim = UsdGeom.Xform.Define(stage, path)
    prim.GetPrim().GetReferences().AddReference(str(asset))
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
    xf.AddRotateZOp().Set(turn)
    xf.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
    prim.GetPrim().SetInstanceable(instanceable)
    return prim


def _prop_source(stage, path: str, usd: Path, keep: str | None) -> None:
    """A class prim for a Poly Haven model that many instances share: the part ``keep`` of its set,
    or the whole set, centered on the origin, rendering through its preview surfaces.

    Kit renders a material's MaterialX surface over its UsdPreviewSurface when both are present,
    and Poly Haven's MaterialX foliage draws pale and ghostly, so this cuts the MaterialX output. A
    textured opacity with no threshold renders as see-through blending, not the solid cutout an
    artist draws a needle or leaf card as, so each gets one.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    asset = Usd.Stage.Open(str(usd))  # held: a prim expires with its stage
    root = asset.GetDefaultPrim()
    part = root.GetChild(keep) if keep else root
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    centre = bounds.ComputeWorldBound(part).ComputeAlignedRange().GetMidpoint()
    stage.CreateClassPrim(path)
    inner = UsdGeom.Xform.Define(stage, f"{path}/Asset")
    inner.GetPrim().GetReferences().AddReference(str(usd))
    UsdGeom.Xformable(inner).AddTranslateOp().Set(Gf.Vec3d(-centre[0], -centre[1], 0.0))
    for child in [c.GetName() for c in root.GetChildren()]:
        if keep and child not in (keep, "_materials"):
            stage.OverridePrim(f"{path}/Asset/{child}").SetActive(False)
    for prim in asset.Traverse():
        here = f"{path}/Asset" + str(prim.GetPath())[len(str(root.GetPath())) :]
        material = UsdShade.Material(prim)
        if material and material.GetSurfaceOutput("mtlx"):
            # Disconnect, not ClearSources: only a disconnect outweighs the asset's own connection.
            UsdShade.Material(stage.OverridePrim(here)).CreateSurfaceOutput("mtlx").DisconnectSource()
        shader = UsdShade.Shader(prim)
        for put in shader.GetInputs() if shader else []:
            # Poly Haven's file lists can ship a texture in another format than its USD names, as
            # rock_moss_set_02's 1k roughness: point every texture input, the MaterialX graph's
            # too, at the file that did arrive, or the packager fails on the missing one.
            file = put.Get()
            if not isinstance(file, Sdf.AssetPath) or not file.path:
                continue
            named = (usd.parent / file.path).resolve()
            if not named.exists():
                found = sorted(named.parent.glob(named.stem + ".*"))
                if not found:
                    raise FileNotFoundError(f"{usd.name} names {file.path}, which Poly Haven did not ship")
                over = UsdShade.Shader(stage.OverridePrim(here))
                over.CreateInput(put.GetBaseName(), Sdf.ValueTypeNames.Asset).Set(str(found[0]))
        if shader and shader.GetIdAttr().Get() == "UsdPreviewSurface":
            opacity = shader.GetInput("opacity")
            if opacity and opacity.HasConnectedSource():
                over = UsdShade.Shader(stage.OverridePrim(here))
                over.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(NEEDLE_CUTOFF)


def _instance(stage, path: str, source: str, at, turn: float, scale: float) -> None:
    """An instance of the source prim ``/World/_Sources/<source>`` at ``at``, turned and scaled."""
    from pxr import Gf, UsdGeom

    prim = UsdGeom.Xform.Define(stage, path)
    prim.GetPrim().GetReferences().AddInternalReference(f"/World/_Sources/{source}")
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(*at))
    xf.AddRotateZOp().Set(turn)
    xf.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
    prim.GetPrim().SetInstanceable(True)


def _sunset_sky(stage, root: str, hdr: Path) -> None:
    """The sky image on a dome, and a low warm sun along its brightest point for crisp shadows."""
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    dome = UsdLux.DomeLight.Define(stage, f"{root}/EnvSky")
    dome.CreateIntensityAttr(SKY_INTENSITY)
    dome.CreateTextureFileAttr(Sdf.AssetPath(str(hdr)))
    dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
    # No turn: Kit maps a lat-long image with the stage's up axis already, and scene_root's 90 degree
    # turn, harmless on its plain dome, stands this image's horizon on end.
    az, el = math.radians(SUN_AZIMUTH_DEG), math.radians(SUN_ELEVATION_DEG)
    to_sun = Gf.Vec3d(-math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el))
    sun = UsdLux.DistantLight.Define(stage, f"{root}/EnvSun")
    sun.CreateIntensityAttr(SUN_INTENSITY)
    sun.CreateColorAttr(Gf.Vec3f(*SUN_COLOR))
    sun.CreateAngleAttr(0.53)
    # A distant light shines down its -z axis: turn its +z towards the sun.
    rot = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), to_sun)
    UsdGeom.Xformable(sun.GetPrim()).AddOrientOp().Set(Gf.Quatf(rot.GetQuat()))


def _pole_frames():
    """Each pole's base point and the direction the line runs through it: along its span at the
    line's ends, halfway between its two spans at a turn.
    """
    frames = []
    for i, (x, y) in enumerate(POLES):
        dirs = []
        for (ax, ay), (bx, by) in [(POLES[j], POLES[j + 1]) for j in (i - 1, i) if 0 <= j < len(POLES) - 1]:
            length = math.hypot(bx - ax, by - ay)
            dirs.append(((bx - ax) / length, (by - ay) / length))
        dx, dy = sum(d[0] for d in dirs), sum(d[1] for d in dirs)
        length = math.hypot(dx, dy)
        frames.append(((x, y, height(x, y)), (dx / length, dy / length)))
    return frames


def author(out_path: str) -> str:
    import tempfile

    from pxr import Sdf, Usd, UsdGeom, UsdUtils

    out_path = str(out_path)
    if not out_path.endswith(".usdz"):
        raise ValueError(f"scene assets are ONE self-contained .usdz, got {out_path!r}")

    poles_usd = fetch_model("modular_electricity_poles")
    props = {name: fetch_model(name, res) for name, _, res, *_ in PROPS}
    props[ROOTS[0]] = fetch_model(ROOTS[0], PROP_RES)
    road = fetch_texture("asphalt_track")
    floor = fetch_texture(FLOOR[0])
    sky = fetch_sky(SKY)

    tmp_usd = tempfile.mktemp(suffix=".usdc")
    stage = Usd.Stage.CreateNew(tmp_usd)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    _sunset_sky(stage, "/World", sky)

    # The ground, the raised tarmac, and its markings a few millimetres over the tarmac, so no two
    # surfaces fight for the same pixels.
    _ground(stage, "/World/Ground/Floor", _textured_material(stage, "/World/Looks/Floor", floor, FLOOR[1]))
    centre = road_centre()
    road_mat = _textured_material(stage, "/World/Looks/Road", road, tile=2.0)
    paint = _plain_material(stage, "/World/Looks/Paint", (0.85, 0.85, 0.8), roughness=0.7)
    wire_mat = _plain_material(stage, "/World/Looks/Wire", (0.08, 0.08, 0.08), metallic=0.6, roughness=0.45)
    _slab(stage, "/World/Ground/Road", _offset(centre, ROAD_HALF_WIDTH), _offset(centre, -ROAD_HALF_WIDTH), road_mat)
    for name, d in (("edge_left", ROAD_HALF_WIDTH - EDGE_INSET), ("edge_right", -(ROAD_HALF_WIDTH - EDGE_INSET))):
        _strip(
            stage,
            f"/World/Ground/Markings/{name}",
            _offset(centre, d + LINE_WIDTH / 2),
            _offset(centre, d - LINE_WIDTH / 2),
            ROAD_THICK + 0.005,
            paint,
        )
    along, dash, n = 0.0, [], 0
    for i, p in enumerate(centre):
        if i:
            along += math.hypot(p[0] - centre[i - 1][0], p[1] - centre[i - 1][1])
        if along % (DASH + GAP) < DASH:
            dash.append(p)
        elif len(dash) > 1:
            _strip(
                stage,
                f"/World/Ground/Markings/dash_{n}",
                _offset(dash, LINE_WIDTH / 2),
                _offset(dash, -LINE_WIDTH / 2),
                ROAD_THICK + 0.005,
                paint,
            )
            dash, n = [], n + 1
        else:
            dash = []

    # Each pole references the whole asset, so its material bindings resolve, with every part but
    # the line pole preset switched off, and turns so its crossarms lie across the line.
    asset = Usd.Stage.Open(str(poles_usd))  # held: a prim expires with its stage
    parts = [p.GetName() for p in asset.GetDefaultPrim().GetChildren()]
    others = [name for name in parts if not name.startswith(POLE_PRESET + "_") and name != "_materials"]
    frames = _pole_frames()
    for i, ((x, y, z), (dx, dy)) in enumerate(frames):
        turn = math.degrees(math.atan2(dy, dx)) + 90.0  # asset x, the crossarm, onto the line's left
        c, s = math.cos(math.radians(turn)), math.sin(math.radians(turn))
        _place(
            stage,
            f"/World/Poles/pole_{i}",
            poles_usd,
            x - POLE_AXIS_X * c,
            y - POLE_AXIS_X * s,
            z,
            turn,
            1.0,
            instanceable=False,
        )
        for name in others:
            stage.OverridePrim(f"/World/Poles/pole_{i}/{name}").SetActive(False)
        _collider_capsule(stage, f"/World/Colliders/pole_{i}", (x, y, z), (x, y, z + POLE_HEIGHT), POLE_RADIUS)
        for j, (length, h) in enumerate(CROSSARMS):
            a = (x + dy * length / 2, y - dx * length / 2, z + h)
            b = (x - dy * length / 2, y + dx * length / 2, z + h)
            _collider_capsule(stage, f"/World/Colliders/pole_{i}_arm_{j}", a, b, 0.06)

    # The wires: from one pole's forward insulator tip to the next pole's rear one, a sagging curve
    # per conductor per span, with a few capsules along each for the physics.
    def tip(frame, side: float, across: float, h: float):
        (x, y, z), (dx, dy) = frame
        return (x + side * INSULATOR_REACH * dx - across * dy, y + side * INSULATOR_REACH * dy + across * dx, z + h)

    for k, (across, h) in enumerate(INSULATORS):
        for s in range(len(frames) - 1):
            a, b = tip(frames[s], 1.0, across, h), tip(frames[s + 1], -1.0, across, h)
            sag = WIRE_SAG * (math.dist(a[:2], b[:2]) / 20.0) ** 2

            def at(t: float, a=a, b=b, sag=sag):
                return (
                    a[0] + t * (b[0] - a[0]),
                    a[1] + t * (b[1] - a[1]),
                    a[2] + t * (b[2] - a[2]) - sag * 4.0 * t * (1.0 - t),
                )

            _tube(
                stage, f"/World/Wires/wire_{k}_{s}", [at(n / (WIRE_POINTS - 1)) for n in range(WIRE_POINTS)], wire_mat
            )
            for c in range(WIRE_COLLIDERS):
                _collider_capsule(
                    stage,
                    f"/World/Colliders/wire_{k}_{s}_{c}",
                    at(c / WIRE_COLLIDERS),
                    at((c + 1) / WIRE_COLLIDERS),
                    0.02,
                )
    _ground_colliders(stage, "/World/Colliders")
    _road_colliders(stage, "/World/Colliders", centre)

    # Plants and rocks at seeded random spots, each clear of the road, the line,
    # the start and landing spots and the square's edge; neighbours can overlap a little, as in a
    # stand of trees. Every instance of a model shares one source prim.
    for name, keep, *_ in PROPS:
        _prop_source(stage, f"/World/_Sources/{name}", props[name], keep)
    _prop_source(stage, f"/World/_Sources/{ROOTS[0]}", props[ROOTS[0]], ROOTS[1])
    rng = random.Random(PROP_SEED)
    line = [
        p
        for a, b in itertools.pairwise(POLES)
        for p in ((a[0] + t / 20 * (b[0] - a[0]), a[1] + t / 20 * (b[1] - a[1])) for t in range(21))
    ]
    placed: list[tuple[float, float, float]] = []
    for name, _, _, radius, count, scales, sink in PROPS:
        tries, n = 0, 0
        while n < count:
            tries += 1
            if tries > PROP_TRIES:
                raise RuntimeError(f"only {n} of {count} {name} fit the placement rules")
            scale = rng.uniform(*scales)
            r = radius * scale
            x, y = rng.uniform(-HALF + r, HALF - r), rng.uniform(-HALF + r, HALF - r)
            if (
                _distance_to(centre, x, y) < ROAD_HALF_WIDTH + r + CLEAR
                or _distance_to(line, x, y) < r + CLEAR
                or min(math.dist((x, y), spot) for spot in CLEAR_SPOTS) < r + CLEAR
                or any(math.hypot(x - px, y - py) < (1.0 - OVERLAP) * (r + pr) for px, py, pr in placed)
            ):
                continue
            _instance(
                stage, f"/World/Props/{name}_{n}", name, (x, y, height(x, y) - sink), rng.uniform(0.0, 360.0), scale
            )
            if name == PROPS[0][0]:  # every fir stands on a root flare
                _instance(
                    stage,
                    f"/World/Props/{ROOTS[0]}_{n}",
                    ROOTS[0],
                    (x, y, height(x, y)),
                    rng.uniform(0.0, 360.0),
                    scale,
                )
            placed.append((x, y, r))
            n += 1

    # One flat layer before packaging: the packager rewrites a nested layer's texture paths to its
    # own folders, which the nested layer then can't resolve; flattened, every path sits in one layer.
    flat_usd = tempfile.mktemp(suffix=".usdc")
    stage.Flatten().Export(flat_usd)
    if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(flat_usd), out_path):
        raise RuntimeError(f"usdz packaging failed for {flat_usd}")
    return out_path


if __name__ == "__main__":
    print(author(sys.argv[1] if len(sys.argv) > 1 else "/tmp/powerline.usdz"))
