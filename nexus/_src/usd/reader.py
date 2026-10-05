"""The reader: the applied nexus schemas of a stage, as declarations.

A nexus schema is an applied API schema whose attributes sit in the `nexus:` namespace, whichever plugin
defines it, so a project's own schemas read the same way as the ones nexus ships.
"""

from __future__ import annotations

import re
from importlib.metadata import version
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


def _retired() -> set[str]:
    """Every schema version a registered plugin lists as retired, under `NexusRetiredSchemas` in its `plugInfo.json`."""
    from pxr import Plug

    plugins = Plug.Registry().GetAllPlugins()
    return {name for plugin in plugins for name in plugin.metadata.get("NexusRetiredSchemas", ())}


def _fail_unread_version(prim, applied: list[str], retired: set[str]) -> None:
    """Fail on a schema the prim lists that no plugin defines, when its family is one a plugin defines or has retired.

    OpenUSD drops such a schema from the prim's applied schemas, so without this the prim would build
    without it and say nothing. The check skips a schema of a family no plugin has ever defined.
    """
    from pxr import Usd

    registry = Usd.SchemaRegistry
    listed = prim.GetMetadata("apiSchemas")
    for schema in listed.GetAppliedItems() if listed else ():
        name = schema.partition(":")[0]  # an instance of a schema applied more than once, `CollectionAPI:colliders`
        if schema in applied or registry.FindSchemaInfo(name) is not None:
            continue
        family, _ = registry.ParseSchemaFamilyAndVersionFromIdentifier(name)
        if name in retired or registry.FindSchemaInfosInFamily(family):
            raise ValueError(
                f"{prim.GetPath()}: applies {name}, a schema version that nexus {version('nexus-sim')} does not define"
            )


def read_declarations(usd_path: str | Path) -> list[tuple[str, str, dict]]:
    """One `(prim path, schema, keyword arguments)` per applied nexus schema on the stage's active prims.

    Each attribute the schema defines becomes the keyword of the same name in snake case.

    Raises:
        ValueError: A prim authors a `nexus:` attribute that none of its applied schemas defines, or an
            asset path that resolves to no file; the message names the prim. Or a prim applies a version
            of a schema that no plugin defines, of a family a plugin defines or has retired; the message
            names the prim, the version and the nexus version reading it.
    """
    from pxr import Usd

    registry = Usd.SchemaRegistry()
    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    retired = _retired()
    declarations = []
    # The default predicate skips inactive prims; instance proxies reach into instanced references.
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        applied = prim.GetAppliedSchemas()
        _fail_unread_version(prim, applied, retired)
        defined = set()
        for schema in applied:
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
