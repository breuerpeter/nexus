"""mkdocs hook: render the schema reference from the nexus Universal Scene Description (USD) schema plugin.

Replaces `<!-- schema-reference -->` with every schema the plugin defines: the prim types it applies to,
and each attribute's type, default, units and range. The page comes from the plugin through OpenUSD's
schema registry, the same definitions the reader enforces, so it can't drift from what the build accepts.
Each attribute's doc in `generatedSchema.usda` ends with its `Range:` and `Units:` lines, the convention
Newton's schemas follow.

As with `benchmarks.py`, this lives under `docs/hooks` as a build tool, not content.
"""

from __future__ import annotations

import re

import nexus  # noqa: F401  # registers the schema plugin
from nexus._src.usd import schema_names

_MARKER = "<!-- schema-reference -->"
_FIELD = re.compile(r"^\s*(Range|Units):\s*(.+?)\s*$", re.MULTILINE)


def _attribute_row(definition, name: str) -> str:
    """One table row: the attribute's name, type, default, units, range and description."""
    doc = definition.GetPropertyMetadata(name, "documentation") or ""
    fields = dict(_FIELD.findall(doc))
    description = " ".join(_FIELD.sub("", doc).split())
    fallback = definition.GetAttributeFallbackValue(name)
    default = f"{fallback:g}" if isinstance(fallback, float) else str(fallback)
    type_name = definition.GetSchemaAttributeSpec(name).typeName
    units, value_range = fields.get("Units", ""), fields.get("Range", "")
    return f"| `{name}` | `{type_name}` | `{default}` | {units} | {value_range} | {description} |"


def schema_reference() -> str:
    """The reference page body: one section per schema the plugin defines."""
    from pxr import Usd

    registry = Usd.SchemaRegistry()
    sections = []
    for schema in schema_names():
        definition = registry.FindAppliedAPIPrimDefinition(schema)
        types = registry.GetAPISchemaCanOnlyApplyToTypeNames(schema)
        applies_to = ", ".join(f"`{name}`" for name in types) or "any prim"
        rows = [_attribute_row(definition, name) for name in definition.GetPropertyNames()]
        header = ["| Attribute | Type | Default | Units | Range | Description |", "|---|---|---|---|---|---|"]
        body = [f"## {schema}", "", definition.GetDocumentation(), "", f"Applies to: {applies_to}.", "", *header]
        sections.append("\n".join(body + rows))
    return "\n\n".join(sections) + "\n"


def on_page_markdown(markdown: str, *, page, config, files) -> str:
    """Replace the `<!-- schema-reference -->` marker with the reference."""
    if _MARKER not in markdown:
        return markdown
    return markdown.replace(_MARKER, schema_reference())
