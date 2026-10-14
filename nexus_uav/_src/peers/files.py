"""The files a run's peers read: the vehicle's and the scene's Universal Scene Description (USD) files, and each
layer either stacks, at their host paths.
"""

from __future__ import annotations

from pathlib import Path


def run_files(*files: str | Path | None) -> list[str]:
    """`files`, the vehicle's and the scene's root files, then each layer one of them stacks.

    A run with an override layer opens its vehicle and its scene through root files that stack the layer on
    each asset, so a peer that reads the files reads each root file's layers too. It leaves out a ``None`` file.
    """
    from pxr import Sdf

    roots = [str(f) for f in files if f]
    layers = [Sdf.Layer.FindOrOpen(f) for f in roots]
    return [*roots, *(layer.ComputeAbsolutePath(sub) for layer in layers for sub in layer.subLayerPaths)]
