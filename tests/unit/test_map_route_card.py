"""map_route card — Playwright HTML path + deterministic Pillow fallback (M4)."""

from __future__ import annotations

from pathlib import Path

import pytest
from director_os.vision_fx.cards import CardError, content_digest, render_card
from director_os.vision_fx.map_route import render_map_route

FIELDS = {
    "title": "PURSUIT ROUTE",
    "subtitle": "Reconstructed from dashcam telemetry",
    "route_points": [[0.05, 0.75], [0.3, 0.45], [0.55, 0.55], [0.8, 0.2], [0.95, 0.3]],
    "distance_label": "4.2 mi",
    "duration_label": "6 min",
}


def _render(tmp_path, backend: str):
    out = tmp_path / f"route_{backend}.png"
    result = render_map_route(FIELDS, out, backend=backend)
    assert result == out and out.exists() and out.stat().st_size > 500
    return out


def test_pillow_fallback_is_deterministic(tmp_path):
    a = _render(tmp_path, "pillow")
    b = tmp_path / "again.png"
    render_map_route(FIELDS, b, backend="pillow")
    assert content_digest(a) == content_digest(b)  # cache-key stable


def test_pillow_fallback_rejects_bad_routes(tmp_path):
    for bad in ([], [[0.1, 0.1]], [[0.5, 2.0], [0.2, 0.2]], "nope"):
        with pytest.raises(CardError):
            render_map_route({**FIELDS, "route_points": bad}, tmp_path / "x.png", backend="pillow")
    with pytest.raises(CardError):
        render_map_route(
            {**FIELDS, "route_points": [[0.1, 0.1]] * 13}, tmp_path / "x.png", backend="pillow"
        )


def test_render_card_dispatches_map_route(tmp_path):
    out = render_card("map_route", FIELDS, tmp_path / "via_render_card.png")
    assert out.exists()


def _chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as p:
            return Path(p.chromium.executable_path).exists()
    except Exception:
        return False


def test_playwright_path_when_available(tmp_path):
    if not _chromium_available():
        pytest.skip("chromium not installed (playwright install chromium)")
    out = _render(tmp_path, "playwright")
    # browser path: render twice -> identical bytes (deterministic HTML/CSS)
    b = tmp_path / "again_pw.png"
    render_map_route(FIELDS, b, backend="playwright")
    assert content_digest(out) == content_digest(b)


def test_broken_playwright_falls_back_to_pillow(tmp_path, monkeypatch):
    """A broken browser install degrades gracefully instead of failing the render."""
    import director_os.vision_fx.map_route as mr

    def boom(*a, **k):
        raise RuntimeError("chromium exploded")

    monkeypatch.setattr(mr, "_render_playwright", boom)
    out = tmp_path / "fallback.png"
    render_map_route(FIELDS, out, backend="auto")
    assert out.exists()
