"""DirectorOS vision FX — cards, punch-ins, redaction, spotlight, compute port."""

from director_os.vision_fx.spotlight import (
    SpotlightError,
    clamp_bbox,
    content_digest,
    render_spotlight_overlay,
)

__all__ = [
    "SpotlightError",
    "clamp_bbox",
    "content_digest",
    "render_spotlight_overlay",
]
