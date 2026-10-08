"""The reader: the applied nexus schemas of a stage, as declarations.

A nexus schema is an applied API schema whose attributes sit in the `nexus:` namespace, whichever plugin
defines it, so a project's own schemas read the same way as the ones nexus ships. A component schema states
the role its component fills by including a role schema, such as `NexusSensorRoleAPI`, as a built-in. A
role schema defines no attribute, so the reader skips it. A component schema can also declare a connection,
a relationship `nexus:inputs:<signal>` whose target picks which component's output its component reads.
"""

from __future__ import annotations

import re
from importlib.metadata import version
from pathlib import Path

_NAMESPACE = "nexus:"
# A connection's namespace: the relationship `nexus:inputs:imu` picks the writer of the signal `imu`.
_INPUTS = "nexus:inputs:"

# Each role schema, which a component schema includes as a built-in, and the role it states.
ROLES = {
    "NexusSensorRoleAPI": "sensor",
    "NexusEstimatorRoleAPI": "estimator",
    "NexusControllerRoleAPI": "controller",
    "NexusForceRoleAPI": "force",
}


def roles(schema: str) -> list[str]:
    """The roles `schema` states: each role schema its definition includes, in the order it lists them."""
    from pxr import Usd

    definition = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(schema)
    return [ROLES[name] for name in definition.GetAppliedAPISchemas() if name in ROLES] if definition else []


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


def _check_type(prim, schema: str) -> None:
    """Fail a schema on a prim of a type its plugin doesn't let it apply to.

    A type OpenUSD doesn't know here, such as Kit's `OmniLidar`, matches by its name.

    Raises:
        ValueError: The prim's type is none of the schema's; the message names the prim and the types.
    """
    from pxr import Usd

    allowed = list(Usd.SchemaRegistry.GetAPISchemaCanOnlyApplyToTypeNames(schema))
    if not allowed or prim.GetTypeName() in allowed:
        return
    for name in allowed:
        tf_type = Usd.SchemaRegistry.GetTypeFromSchemaTypeName(name)
        if tf_type and prim.IsA(tf_type):
            return
    raise ValueError(
        f"{prim.GetPath()}: {schema} applies to a prim of type {' or '.join(allowed)}, "
        f"and this prim's type is {prim.GetTypeName() or 'none'}"
    )


def read_declarations(usd_path: str | Path) -> list[tuple[str, str, dict]]:
    """One `(prim path, schema, keyword arguments)` per applied nexus schema on the stage's active prims.

    Each attribute the schema defines becomes the keyword of the same name in snake case. A relationship it
    declares is a connection, which :func:`read_connections` reads.

    Raises:
        ValueError: A prim authors a `nexus:` attribute that none of its applied schemas defines, an
            asset path that resolves to no file, or a schema that doesn't apply to its type; the
            message names the prim. Or a prim applies a version
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
                _check_type(prim, schema)
                defined.update(names)
                kwargs = {
                    _keyword(name): _value(prim, name) for name in names if definition.GetAttributeDefinition(name)
                }
                declarations.append((str(prim.GetPath()), schema, kwargs))
        authored = [prop.GetName() for prop in prim.GetAuthoredPropertiesInNamespace(_NAMESPACE.rstrip(":"))]
        if undefined := [name for name in authored if name not in defined]:
            raise ValueError(f"{prim.GetPath()}: no applied schema defines {', '.join(undefined)}")
    return declarations


def read_connections(usd_path: str | Path) -> dict[str, dict[str, str]]:
    """Each connection the stage's active prims author, by prim: the signal a relationship `nexus:inputs:<signal>`
    names, and the path of the prim it targets, whose component writes that signal.

    The reader of :func:`read_declarations` fails a relationship no applied schema declares, so each
    connection here is one a schema declares.

    Raises:
        ValueError: A connection targets more than one prim; the message names the prim and the connection.
    """
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    connections: dict[str, dict[str, str]] = {}
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        for rel in prim.GetRelationships():
            targets = rel.GetTargets() if rel.GetName().startswith(_INPUTS) else []
            if len(targets) > 1:
                raise ValueError(
                    f"{prim.GetPath()}: {rel.GetName()} targets {len(targets)} prims; a connection names one"
                )
            if targets:
                connections.setdefault(str(prim.GetPath()), {})[rel.GetName().removeprefix(_INPUTS)] = str(targets[0])
    return connections
