"""The reader: the applied nexus schemas of a stage, as declarations.

A nexus schema is an applied API schema whose attributes sit in the `nexus:` namespace, whichever plugin
defines it, so a project's own schemas read the same way as the ones nexus ships.
"""

from __future__ import annotations

import re
from pathlib import Path

_NAMESPACE = "nexus:"


def _keyword(name: str) -> str:
    """The keyword argument for a schema attribute: `nexus:accNoise` becomes `acc_noise`."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name.removeprefix(_NAMESPACE)).lower()


def _value(prim, name: str):
    """The attribute's value; an asset path becomes the file it resolves to, against the layer that authored it."""
    from pxr import Sdf

    value = prim.GetAttribute(name).Get()
    if isinstance(value, Sdf.AssetPath):
        if value.path and not value.resolvedPath:
            raise ValueError(f"{prim.GetPath()}: {name} names {value.path!r}, which resolves to no file")
        return value.resolvedPath or None
    return value


def read_declarations(usd_path: str | Path) -> list[tuple[str, str, dict]]:
    """One `(prim path, schema, keyword arguments)` per applied nexus schema on the stage's active prims.

    Each attribute the schema defines becomes the keyword of the same name in snake case.

    Raises:
        ValueError: A prim authors a `nexus:` attribute that none of its applied schemas defines, or an
            asset path that resolves to no file; the message names the prim.
    """
    from pxr import Usd

    registry = Usd.SchemaRegistry()
    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    declarations = []
    # The default predicate skips inactive prims; instance proxies reach into instanced references.
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        defined = set()
        for schema in prim.GetAppliedSchemas():
            definition = registry.FindAppliedAPIPrimDefinition(schema)
            if definition is None:  # one instance of a schema applied more than once, `CollectionAPI:colliders`
                continue
            names = [name for name in definition.GetPropertyNames() if name.startswith(_NAMESPACE)]
            if names:
                defined.update(names)
                kwargs = {_keyword(name): _value(prim, name) for name in names}
                declarations.append((str(prim.GetPath()), schema, kwargs))
        authored = [prop.GetName() for prop in prim.GetAuthoredPropertiesInNamespace(_NAMESPACE.rstrip(":"))]
        if undefined := [name for name in authored if name not in defined]:
            raise ValueError(f"{prim.GetPath()}: no applied schema defines {', '.join(undefined)}")
    return declarations
