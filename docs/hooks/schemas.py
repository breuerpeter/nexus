"""mkdocs hook: render the schema reference from the nexus Universal Scene Description (USD) schema plugin.

Replaces `<!-- schema-reference -->` with every schema the plugin defines: the prim types it applies to,
the role a component schema states, and each attribute's type, default, units and range. The page comes from the plugin through OpenUSD's
schema registry, the same definitions the reader enforces, so it can't drift from what the build accepts.
Each attribute's doc in `generatedSchema.usda` ends with its `Range:` and `Units:` lines, the convention
Newton's schemas follow.

As with `benchmarks.py`, this lives under `docs/hooks` as a build tool, not content.
"""

from __future__ import annotations

import re

import nexus_sim  # noqa: F401  # registers the schema plugin
from nexus_sim._src.usd import schema_names
from nexus_sim._src.usd.reader import ROLES, roles

_MARKER = "<!-- schema-reference -->"
_FIELD = re.compile(r"^\s*(Range|Units):\s*(.+?)\s*$", re.MULTILINE)


def _attribute_row(definition, name: str) -> str:
    """One table row: the property's name, type, default, units, range and description. A connection, a
    relationship, has no default, units or range.
    """
    doc = definition.GetPropertyMetadata(name, "documentation") or ""
    fields = dict(_FIELD.findall(doc))
    description = " ".join(_FIELD.sub("", doc).split())
    if definition.GetRelationshipDefinition(name):
        return f"| `{name}` | `relationship` | | | | {description} |"
    fallback = definition.GetAttributeFallbackValue(name)
    default = f"{fallback:g}" if isinstance(fallback, float) else str(fallback)
    type_name = definition.GetSchemaAttributeSpec(name).typeName
    units, value_range = fields.get("Units", ""), fields.get("Range", "")
    return f"| `{name}` | `{type_name}` | `{default}` | {units} | {value_range} | {description} |"


def _changes(registry, schema: str) -> list[str]:
    """What `schema` changed from the version of its family before it; nothing for a family's first version."""
    family, version = registry.ParseSchemaFamilyAndVersionFromIdentifier(schema)
    earlier = [info for info in registry.FindSchemaInfosInFamily(family) if info.version < version]
    if not earlier:
        return []
    before = max(earlier, key=lambda info: info.version).identifier
    old, new = registry.FindAppliedAPIPrimDefinition(before), registry.FindAppliedAPIPrimDefinition(schema)
    old_rows = {name: _attribute_row(old, name) for name in old.GetPropertyNames()}
    new_rows = {name: _attribute_row(new, name) for name in new.GetPropertyNames()}
    parts = [f"adds `{name}`" for name in new_rows if name not in old_rows]
    parts += [f"removes `{name}`" for name in old_rows if name not in new_rows]
    parts += [f"changes `{name}`" for name in new_rows if name in old_rows and new_rows[name] != old_rows[name]]
    return [f"Changed from `{before}`: {', '.join(parts) or 'no attribute'}.", ""]


def schema_reference(schemas: list[str] | None = None) -> str:
    """The reference page body: one section per schema in `schemas`, by default every one the plugin defines.

    A later version of a family states what it changed from the version before it.
    """
    from pxr import Usd

    registry = Usd.SchemaRegistry()
    sections = []
    for schema in schema_names() if schemas is None else schemas:
        definition = registry.FindAppliedAPIPrimDefinition(schema)
        types = registry.GetAPISchemaCanOnlyApplyToTypeNames(schema)
        applies_to = ", ".join(f"`{name}`" for name in types) or "any prim"
        # A role schema states no role of its own, and a peer's schema none at all.
        role = "" if schema in ROLES else ", ".join(roles(schema))
        rows = [_attribute_row(definition, name) for name in definition.GetPropertyNames()]
        header = ["| Property | Type | Default | Units | Range | Description |", "|---|---|---|---|---|---|"]
        body = [f"## {schema}", "", definition.GetDocumentation(), "", f"Applies to: {applies_to}.", ""]
        body += [f"Role: {role}.", ""] if role else []
        sections.append("\n".join(body + _changes(registry, schema) + header + rows))
    return "\n\n".join(sections) + "\n"


def on_page_markdown(markdown: str, *, page, config, files) -> str:
    """Replace the `<!-- schema-reference -->` marker with the reference."""
    if _MARKER not in markdown:
        return markdown
    return markdown.replace(_MARKER, schema_reference())
