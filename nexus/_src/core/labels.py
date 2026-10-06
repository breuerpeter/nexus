"""Instance names from model labels: the leaf a recorded row's path and a Recorder key end in."""

from __future__ import annotations

from collections import Counter


def leaf_keys(labels: list[str]) -> list[str]:
    """Friendly instance names from full model labels, often Universal Scene Description (USD) prim paths
    such as ``/astro_max/Geometry/body_frd``: the leaf name after the last ``/`` when it's unique across
    the set, as in ``body_frd`` or ``rotor_1``, else the full label, so a leaf collision never silently
    aliases two entities. The ``sim.physics[...]`` keys are exactly these, and so is the leaf of each
    body's entity path in a recording.
    """
    leaves = [str(lbl).rsplit("/", 1)[-1] for lbl in labels]
    counts = Counter(leaves)
    return [leaf if counts[leaf] == 1 else str(lbl) for lbl, leaf in zip(labels, leaves, strict=True)]
