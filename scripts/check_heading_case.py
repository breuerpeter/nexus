#!/usr/bin/env python
"""Enforce sentence-case Markdown headings across the docs.

Sentence case = only the first word is capitalized; every later word is lowercase unless it is a
proper noun or acronym. Acronyms and CamelCase names (``PX4``, ``MAVLink``, ``NMPC``) are allowed
automatically; ordinary proper nouns (``Newton``, ``Astro``) are listed in ``EXCEPTIONS``.

Usage: ``python scripts/check_heading_case.py docs/**/*.md`` (or pass files; pre-commit passes them).
Exits non-zero and prints ``file:line`` for each offending heading. Tune the lists below as the
vocabulary grows.
"""

from __future__ import annotations

import re
import sys

# Proper nouns allowed to stay capitalized mid-heading: case-sensitive, single-initial-cap words
# that the acronym and mixed-case heuristics below won't catch on their own.
EXCEPTIONS = {
    "Newton",
    "Warp",
    "Omniverse",
    "Kit",
    "Isaac",
    "Lab",
    "Sim",
    "Astro",
    "Max",
    "Pilot",
    "Pro",
    "Freefly",
    "Systems",
    "Pegasus",
    "Simulator",
    "Project",
    "Rerun",
    "Cesium",
    "Quad",
    "Docker",
    "Python",
    "Linux",
    "Gazebo",
    "Apache",
    "Paraverse",
    "Suite",
    "App",
    "X",
    # products / codenames / platforms that appear in the planning & spec docs
    "Node",
    "Web",
    "Android",
    "Ubuntu",
    "GitHub",
    # months are proper nouns
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
}
# Proper nouns intentionally lowercase, allowed as the first word: tool and package names.
LOWER_OK = {"acados", "release", "ffmpeg", "mkdocs"}

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
_CODE_SPAN = re.compile(r"`[^`]*`")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_ATTR = re.compile(r"\{[^}]*\}\s*$")  # trailing {#id .class} attr-list
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")
# A leading list label that isn't a word: "A.", "4.", "9.3", "11.", "A1", "Plane 1",
# "Task 6:", "Path A". The single-letter or single-digit label needs a separator, a space, or the
# end after it, so the check doesn't mis-split a real two-word title such as "File Structure" into a
# bogus "File S" label.
_ENUM = re.compile(
    r"^\s*(?:[A-Za-z]?\d+(?:\.\d+)*\.?|[A-Z]\.|[A-Z][a-z]+\s+(?:[A-Z]|\d+)(?=[\s–:.\-]|$))\s*[–\-:.]?\s*"
)


def _auto_ok(token: str) -> bool:
    """True if the token is an acronym, has mixed case, or has a digit.

    Examples: PX4, MAVLink, Nonlinear Model Predictive Control (NMPC).
    """
    return (
        any(c.isdigit() for c in token)
        or sum(c.isupper() for c in token) >= 2
        or any(c.isupper() for c in token[1:])
        or token in EXCEPTIONS
    )


def offending_words(text: str) -> list[str]:
    """Return the words in a heading that violate sentence case; an empty list means compliant."""
    text = _ATTR.sub("", text)
    text = _ENUM.sub("", text, count=1)  # drop a leading list label (A. / 4. / A1 / Plane 1)
    starts_with_code = text.lstrip().startswith("`")  # then the code sets the first word's case
    text = _CODE_SPAN.sub("", text)  # drop inline code: filenames, identifiers
    text = _LINK.sub(r"\1", text)  # keep link text, drop the URL
    words = _WORD.findall(text)
    if not words:
        return []
    bad = []
    first, *rest = words
    if not starts_with_code and first[0].islower() and not _auto_ok(first) and first not in LOWER_OK:
        bad.append(first)
    bad += [w for w in rest if w[0].isupper() and not _auto_ok(w)]
    return bad


def check_file(path: str) -> list[tuple[int, str, list[str]]]:
    out = []
    in_fence = False
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.lstrip().startswith(("```", "~~~")):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            m = _HEADING.match(line)
            if not m:
                continue
            bad = offending_words(m.group(2))
            if bad:
                out.append((n, m.group(2), bad))
    return out


def main(argv: list[str]) -> int:
    failures = 0
    for path in argv:
        if not path.endswith(".md"):
            continue
        for n, heading, bad in check_file(path):
            failures += 1
            print(f"{path}:{n}: non-sentence-case heading {bad}: {heading!r}")
    if failures:
        print(
            f"\n{failures} heading(s) not in sentence case. Lowercase the flagged words, or add "
            f"genuine proper nouns to EXCEPTIONS in scripts/check_heading_case.py."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
