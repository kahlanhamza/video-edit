"""Guardrail 8 (PLAN §9.8): licensing safety at scale — no Remotion anywhere."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _pyprojects() -> list[Path]:
    return sorted(REPO_ROOT.rglob("pyproject.toml"))


@pytest.mark.parametrize("pyproject", _pyprojects(), ids=lambda p: str(p.parent.name))
def test_no_remotion_dependency(pyproject: Path):
    text = pyproject.read_text().lower()
    assert "remotion" not in text, (
        f"{pyproject}: Remotion's Company License triggers above 3 seats — "
        "use FFmpeg + Pillow + Playwright or Apache-2.0 HyperFrames"
    )


def test_docker_compose_has_no_remotion_service():
    compose = (REPO_ROOT / "docker-compose.yml").read_text().lower()
    assert "remotion" not in compose


def test_approved_render_stack_present():
    vision_fx = (REPO_ROOT / "packages" / "vision_fx" / "pyproject.toml").read_text().lower()
    assert "pillow" in vision_fx
    otio_engine = (REPO_ROOT / "packages" / "otio_engine" / "pyproject.toml").read_text().lower()
    assert "opentimelineio" in otio_engine
