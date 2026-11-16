"""vision_fx: collision-free cards + blur filter chains."""

from __future__ import annotations

import pytest
from director_os.core.models import BBox, BlurRegion
from director_os.vision_fx.cards import (
    CardError,
    pick_corner,
    render_card,
)
from director_os.vision_fx.redact import build_blur_chain, validate_regions
from PIL import Image


def test_pick_corner_avoids_occupied_tr():
    # 280x150 boxes leave room for distinct corners on a 640x360 canvas
    occupied = [(640 - 280 - 24, 24, 280, 150)]  # top-right taken
    x, y = pick_corner(640, 360, 280, 150, occupied)
    assert (x, y) == (24, 24)  # top-left free


def test_pick_corner_stacks_when_all_corners_full():
    m, cw, ch, bw, bh = 24, 640, 360, 280, 150
    occupied = [
        (m, m, bw, bh),
        (cw - bw - m, m, bw, bh),
        (m, ch - bh - m, bw, bh),
        (cw - bw - m, ch - bh - m, bw, bh),
    ]
    x, y = pick_corner(cw, ch, bw, bh, occupied)
    assert x >= 0 and y >= 0 and x + bw <= cw and y + bh <= ch


def test_render_card_location_intro(tmp_path):
    out = render_card(
        "location_intro",
        {"title": "JACKSONVILLE SHERIFF'S OFFICE", "subtitle": "Traffic stop escalates"},
        tmp_path / "card.png",
    )
    with Image.open(out) as im:
        assert im.size == (640, 180)
        assert im.mode == "RGBA"


def test_unknown_and_m4_templates_rejected(tmp_path):
    with pytest.raises(CardError):
        render_card("nope", {}, tmp_path / "x.png")
    with pytest.raises(CardError):
        render_card("map_route", {}, tmp_path / "y.png")


def test_blur_chain_two_regions():
    regions = [
        BlurRegion(bbox=BBox(x=10, y=10, w=100, h=60), sigma=20),
        BlurRegion(bbox=BBox(x=400, y=200, w=120, h=80), sigma=12),
    ]
    frag = build_blur_chain("[v0]", "[vblur]", regions, 640, 360)
    assert "split=3" in frag
    assert frag.count("boxblur") == 2
    assert frag.count("overlay") == 2
    assert frag.endswith("[vblur]") or "[vblur]" in frag


def test_blur_chain_clamps_sigma_to_region_size():
    regions = [BlurRegion(bbox=BBox(x=0, y=0, w=8, h=6), sigma=200)]
    frag = build_blur_chain("[v0]", "[vblur]", regions, 640, 360)
    # radius must be <= (min(w,h)-1)//2 = 2
    assert "luma_radius=2" in frag


def test_out_of_frame_regions_pass_through():
    regions = [BlurRegion(bbox=BBox(x=1000, y=900, w=50, h=50))]
    frag = build_blur_chain("[v0]", "[vout]", regions, 640, 360)
    assert "null" in frag and "boxblur" not in frag
    assert validate_regions(regions, 640, 360) == 0
