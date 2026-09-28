"""mkdocs hook: render the schema reference from the nexus Universal Scene Description (USD) schema plugin.

Replaces `<!-- schema-reference -->` with every schema the plugin defines: what it applies to, its
attributes, their units, limits and defaults. The page comes from the plugin, so it can't drift from
what the reader accepts.
"""

from __future__ import annotations


def schema_reference() -> str:
    """The reference page body, rendered from the plugin."""
    ...
