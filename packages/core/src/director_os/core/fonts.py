"""Exact-upright font resolution — PLAN guardrail 7.

Substring font matching is a trap: searching "Roboto-Bold" can match
`Roboto-BoldItalic.ttf` and produce slanted captions. Fonts are therefore matched
against exact upright filenames only.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_SEARCH_DIRS = [
    "/usr/share/fonts/truetype",
    "/usr/local/share/fonts",
    str(Path.home() / ".fonts"),
]
FALLBACK_FILE = "DejaVuSans-Bold.ttf"


class FontNotFoundError(RuntimeError):
    pass


def is_upright_bold(filename: str) -> bool:
    """True only for exact upright Bold faces — rejects Italic/Oblique look-alikes."""
    name = Path(filename).name
    return name.endswith("-Bold.ttf") and "Italic" not in name and "Oblique" not in name


def pick_font(family: str = "Roboto", search_dirs: list[str] | None = None) -> Path:
    """Find `<FamilyNormalized>-Bold.ttf` exactly; fall back to DejaVuSans-Bold."""
    dirs = [Path(d) for d in (search_dirs or DEFAULT_SEARCH_DIRS)]
    target = family.replace(" ", "") + "-Bold.ttf"
    for d in dirs:
        if not d.exists():
            continue
        for p in d.rglob("*.ttf"):
            if p.name == target and is_upright_bold(p.name):
                return p
    for d in dirs:
        if not d.exists():
            continue
        for p in d.rglob(FALLBACK_FILE):
            if p.name == FALLBACK_FILE:
                return p
    raise FontNotFoundError(
        f"no upright bold font for family '{family}' and no {FALLBACK_FILE} fallback "
        f"under: {[str(d) for d in dirs]}"
    )
