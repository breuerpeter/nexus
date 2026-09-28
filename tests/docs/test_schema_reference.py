"""The schema reference page, which the docs hook renders from the nexus Universal Scene Description (USD) schema plugin."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _hook():
    spec = importlib.util.spec_from_file_location("schemas", ROOT / "docs" / "hooks" / "schemas.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_reference_page_is_generated_from_the_plugin():
    """The docs reference page comes from the plugin: every schema, what it applies to, its
    attributes, their units, limits and defaults.

    Given the plugin, when the hook renders the page, then it carries each schema, the prim types it applies
    to, and each attribute with its unit, limit and default.
    """
    import nexus  # noqa: F401  # registers the plugin
    from pxr import Usd

    from nexus._src.usd import schema_names

    page = _hook().schema_reference()
    registry = Usd.SchemaRegistry()
    expected = []
    for schema in schema_names():
        expected.append(schema)
        expected += registry.GetAPISchemaCanOnlyApplyToTypeNames(schema)
        definition = registry.FindAppliedAPIPrimDefinition(schema)
        for name in definition.GetPropertyNames():
            expected.append(name)
            fallback = definition.GetAttributeFallbackValue(name)
            if isinstance(fallback, (int, float)):
                expected.append(f"{fallback:g}")
            expected += re.findall(r"(?:Units|Range): (.+)", definition.GetPropertyMetadata(name, "documentation"))
    assert expected and [text for text in expected if text not in page] == []
