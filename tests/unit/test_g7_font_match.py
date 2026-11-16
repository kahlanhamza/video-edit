"""Guardrail 7 (PLAN §9.7): exact upright font filenames — never BoldItalic."""

from __future__ import annotations

import pytest
from director_os.core.fonts import FontNotFoundError, is_upright_bold, pick_font


def make_tree(tmp_path, files):
    for rel in files:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"ttf")
    return tmp_path


def test_upright_bold_predicate_rejects_italic_trap():
    assert is_upright_bold("Roboto-Bold.ttf")
    assert not is_upright_bold("Roboto-BoldItalic.ttf")
    assert not is_upright_bold("Roboto-Italic.ttf")
    assert not is_upright_bold("Roboto-BoldOblique.ttf")
    assert not is_upright_bold("Roboto-Regular.ttf")


def test_picks_exact_upright_family(tmp_path):
    tree = make_tree(
        tmp_path,
        [
            "roboto/Roboto-BoldItalic.ttf",
            "roboto/Roboto-Bold.ttf",
            "roboto/Roboto-Regular.ttf",
        ],
    )
    assert pick_font("Roboto", search_dirs=[str(tree)]).name == "Roboto-Bold.ttf"


def test_falls_back_to_dejavu(tmp_path):
    tree = make_tree(tmp_path, ["dejavu/DejaVuSans-Bold.ttf"])
    assert pick_font("Roboto", search_dirs=[str(tree)]).name == "DejaVuSans-Bold.ttf"


def test_missing_fonts_raise_loudly(tmp_path):
    empty = tmp_path / "none"
    empty.mkdir()
    with pytest.raises(FontNotFoundError):
        pick_font("Roboto", search_dirs=[str(empty)])
