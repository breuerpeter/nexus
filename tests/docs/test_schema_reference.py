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


def test_the_hook_replaces_its_marker_with_the_reference():
    """The hook replaces the `<!-- schema-reference -->` marker with the rendered reference."""
    page = _hook().on_page_markdown("Intro.\n\n<!-- schema-reference -->\n", page=None, config=None, files=None)
    assert "<!-- schema-reference -->" not in page and "## NexusImuAPI" in page


def test_the_reference_page_lists_every_version_of_a_family_and_what_changed():
    """The schema reference page lists every version of a family and what changed in each.

    Given a plugin that defines a family at two versions with a renamed attribute, when the hook renders
    the page, then it carries a section per version and the later one names the attribute that changed.
    The family is the stand-in project's `StandInGearAPI`, whose second version renames `nexus:ratio`.
    """
    page = _hook().schema_reference(["StandInGearAPI", "StandInGearAPI_1"])
    first, _, second = page.partition("## StandInGearAPI_1\n")
    assert "## StandInGearAPI\n" in first and "nexus:ratio" in second


def test_the_reference_page_is_generated_from_the_plugin():
    """The docs reference page comes from the plugin: every schema, what it applies to, its
    attributes, their units, limits and defaults.

    Given the plugin, when the hook renders the page, then it carries each schema, the prim types it applies
    to, and each attribute with its unit, limit and default.
    """
    import nexus_sim  # noqa: F401  # registers the plugin
    from pxr import Usd

    from nexus_sim._src.usd import schema_names

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
