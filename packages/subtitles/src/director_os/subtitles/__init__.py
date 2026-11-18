"""DirectorOS subtitles — alignment + 2-row ASS captions (PLAN §3.3F, §4.6)."""

from director_os.core.fonts import is_upright_bold, pick_font  # re-export (guardrail 7)
from director_os.subtitles.proofread import (
    JARGON_PHRASES,
    JARGON_SINGLE,
    ProofreadError,
    prepare_transcript,
    proofread_words,
    tag_speakers,
)

__all__ = [
    "JARGON_PHRASES",
    "JARGON_SINGLE",
    "ProofreadError",
    "is_upright_bold",
    "pick_font",
    "prepare_transcript",
    "proofread_words",
    "tag_speakers",
]
