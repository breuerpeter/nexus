"""``scripts/assets/scene_root.py`` puts a converted scene layer under one root prim, the default
prim, with the sky beneath it. A re-root keeps every mesh bound to its material and every shader
network connected.
"""

import importlib.util
import pathlib

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

ROOT = pathlib.Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location("scene_root", ROOT / "scripts" / "assets" / "scene_root.py")
scene_root = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scene_root)


def _bound_mesh(stage, mesh_path, material_path):
    """Define a mesh bound to a material whose preview surface reads its color from a texture."""
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    material = UsdShade.Material.Define(stage, material_path)
    surface = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    surface.CreateIdAttr("UsdPreviewSurface")
    texture = UsdShade.Shader.Define(stage, f"{material_path}/Tex")
    texture.CreateIdAttr("UsdUVTexture")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def _converted_scan(path, *, up_axis_op=True):
    """A layer in the shape of the OBJ converter's output: a ``/World`` holding two bound meshes, with
    the converter's up-axis rotate op on ``/World`` unless ``up_axis_op`` is false.
    """
    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World")
    if up_axis_op:
        world.AddRotateXOp().Set(90.0)
    _bound_mesh(stage, "/World/site/mesh", "/World/Looks/ground")
    _bound_mesh(stage, "/World/site/mesh1", "/World/Looks/box")
    stage.Save()
    return str(path)


def _bindings(path):
    stage = Usd.Stage.Open(path)
    return {
        str(p.GetPath()): [str(t) for t in UsdShade.MaterialBindingAPI(p).GetDirectBindingRel().GetTargets()]
        for p in stage.Traverse()
        if p.IsA(UsdGeom.Mesh)
    }


def test_a_rerooted_mesh_stays_bound_to_its_material(tmp_path):
    """A re-rooted mesh stays bound to its material: each mesh's binding targets the same
    ``Material`` prim at its new path, one the stage holds.
    """
    path = _converted_scan(tmp_path / "scene.usda")

    scene_root.finalize_scene_layer(path)

    assert _bindings(path) == {
        "/Scene/World/site/mesh": ["/Scene/World/Looks/ground"],
        "/Scene/World/site/mesh1": ["/Scene/World/Looks/box"],
    }


def test_a_rerooted_material_keeps_its_shader_network_connected(tmp_path):
    """A re-rooted material keeps its shader network connected: every connection resolves to an
    attribute the stage holds.
    """
    path = _converted_scan(tmp_path / "scene.usda")

    scene_root.finalize_scene_layer(path)

    stage = Usd.Stage.Open(path)
    dangling = [
        (str(a.GetPath()), str(src))
        for p in stage.Traverse()
        for a in p.GetAttributes()
        for src in a.GetConnections()
        if not stage.GetAttributeAtPath(src)
    ]
    assert dangling == []


def test_a_clean_single_root_keeps_its_paths_and_bindings(tmp_path):
    """A layer with one transform-free root ``Xform`` keeps its paths and bindings: no prim moves,
    the binding stays, ``/World`` is the default prim and holds the sky.
    """
    path = _converted_scan(tmp_path / "scene.usda", up_axis_op=False)

    scene_root.finalize_scene_layer(path)

    stage = Usd.Stage.Open(path)
    assert (
        str(stage.GetDefaultPrim().GetPath()),
        [str(p.GetPath()) for p in stage.GetPseudoRoot().GetChildren()],
        _bindings(path),
        [str(p.GetPath()) for p in stage.GetPrimAtPath("/World").GetChildren() if p.GetName().startswith("Env")],
    ) == (
        "/World",
        ["/World"],
        {"/World/site/mesh": ["/World/Looks/ground"], "/World/site/mesh1": ["/World/Looks/box"]},
        ["/World/EnvSky", "/World/EnvSun"],
    )


def test_a_splat_typed_root_moves_under_a_fresh_root_xform(tmp_path):
    """A splat's typed root still moves under a fresh root ``Xform``: that prim sits under a new
    ``/World`` ``Xform``, which is the default prim and holds the sky.
    """
    path = str(tmp_path / "splat.usda")
    stage = Usd.Stage.CreateNew(path)
    splat = stage.DefinePrim("/GaussianSplat", "ParticleField3DGaussianSplat")
    UsdGeom.Xformable(splat).AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    stage.Save()

    scene_root.finalize_scene_layer(path)

    stage = Usd.Stage.Open(path)
    world = stage.GetDefaultPrim()
    assert (
        str(world.GetPath()),
        world.GetTypeName(),
        [(str(p.GetPath()), p.GetTypeName()) for p in world.GetChildren()],
    ) == (
        "/World",
        "Xform",
        [
            ("/World/GaussianSplat", "ParticleField3DGaussianSplat"),
            ("/World/EnvSky", "DomeLight"),
            ("/World/EnvSun", "DistantLight"),
        ],
    )
