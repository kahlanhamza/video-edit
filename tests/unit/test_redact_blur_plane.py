"""The blur-plane law: boxblur radius rides the EVEN crop plane (yuv420p).

Tape-1 wire fact 2026-11-25: a 43px tracked face region computed luma_radius
21, but the yuv420p crop plane rounded down to 42 — ffmpeg refuses
"radius 21, must be >= 0 and < 21" and the whole segment render died.
"""

from director_os.core.models import BBox, BlurRegion
from director_os.vision_fx.redact import _safe_sigmas, build_blur_chain


def test_radius_clamps_to_the_even_plane():
    # 43px region: the plane is 42 -> radius must be < 21
    assert _safe_sigmas(21, 43, 43) == (20, 10)
    assert _safe_sigmas(21, 42, 42) == (20, 10)
    assert _safe_sigmas(50, 43, 43)[0] == 20  # never above the plane law
    assert _safe_sigmas(3, 3, 3) == (0, 0)  # sub-even region: no blur possible


def test_blur_chain_never_emits_an_illegal_radius():
    regions = [
        BlurRegion(bbox=BBox(x=10, y=10, w=43, h=43), sigma=21),
        BlurRegion(bbox=BBox(x=200, y=120, w=41, h=37), sigma=18),
    ]
    cmd = build_blur_chain("[v0]", "[vout]", regions, 1920, 1080)
    for token in cmd.replace(";", ":").split(":"):
        if "luma_radius" in token:
            r = int(token.split("=")[1])
            assert r < 21, token
