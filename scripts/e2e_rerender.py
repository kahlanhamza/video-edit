"""Re-run the pipeline on an existing project (crash-resume semantics).

The persisted transcript means perception is skipped (no ASR spend); planning
reuses the active timeline; render re-runs only segments whose hash changed —
which, after drawing a blur box in the studio, is every raw segment that now
carries the project-level redaction.
"""

from __future__ import annotations

import sys

from director_os.core.config import get_settings
from director_os.otio_engine.segment_cache import VideoRenderSpec
from director_os.worker.pipeline import make_deps, run_pipeline

pid = sys.argv[1]
src = sys.argv[2] if len(sys.argv) > 2 else "output/e2e/fixture_speech.mp4"

settings = get_settings()
deps = make_deps(
    settings.database_url,
    transcribe=None,  # transcript already persisted — stage skips before needing it
    video=VideoRenderSpec(w=640, h=360, fps=30),
    log=print,
)
result = run_pipeline(src, pid, deps)
print("\nre-render:", result.render.rendered, "rendered /", result.render.cached, "cached")
print("audit ok:", result.audit.ok)
print("final:", result.final_path)
