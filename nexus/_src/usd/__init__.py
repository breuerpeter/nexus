"""The nexus Universal Scene Description (USD) schema plugin and its reader.

The plugin is codeless: `plugInfo.json` and `generatedSchema.usda` beside this file, and no compiled code.
"""

from __future__ import annotations

from pathlib import Path

_PLUGIN = Path(__file__).parent


def register_plugins() -> None:
    """Register the nexus schema plugin with OpenUSD.

    OpenUSD builds its schema registry once, on first use, and a plugin registered after that never shows
    in it, so `import nexus` calls this before any caller can open a stage.
    """
    from pxr import Plug

    Plug.Registry().RegisterPlugins([str(_PLUGIN)])


def schema_names() -> tuple[str, ...]:
    """The identifiers of every applied API schema the nexus plugin defines."""
    ...
