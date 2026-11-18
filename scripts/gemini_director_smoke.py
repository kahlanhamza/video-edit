"""Live Gemini smoke: build_backend() with blank LLM_BASE_URL -> GeminiRestBackend.

Drives one real propose() against a 3-clip v1 cut and prints what the model
wants to do. Exercises: AQ-key auth (header + ?key= fallback), function
declarations from the Pydantic registry, thoughtSignature capture.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "apps/worker/src"))

from director_os.agent_runtime.agents import ScriptedBackend  # noqa: E402, F401 (port shape check)
from director_os.agent_runtime.checkpoint import CheckpointStore  # noqa: E402, F401
from director_os.agent_runtime.editing_tools import DirectorToolbox, build_director_registry  # noqa: E402
from director_os.agent_runtime.llm import build_backend  # noqa: E402
from director_os.core.db import init_models, make_engine, make_session_factory  # noqa: E402
from director_os.otio_engine.contract import RawClipMeta  # noqa: E402
from director_os.otio_engine.timeline import (  # noqa: E402
    MASTER_TRACK,
    OVERLAY_TRACK,
    ClipSpec,
    GapSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
)

PROJECT = "gemini_smoke"

timeline = build_timeline(
    TimelineSpec(
        name="smoke",
        fps=30,
        width=1280,
        height=720,
        tracks=[
            TrackSpec(
                name=MASTER_TRACK,
                items=[
                    ClipSpec(
                        name="raw_1",
                        source_start_s=0.0,
                        dur_s=12.0,
                        meta=RawClipMeta(asset_id="a1"),
                    ),
                    ClipSpec(
                        name="raw_2",
                        source_start_s=12.0,
                        dur_s=8.0,
                        meta=RawClipMeta(asset_id="a1"),
                    ),
                    ClipSpec(
                        name="raw_3",
                        source_start_s=20.0,
                        dur_s=5.0,
                        meta=RawClipMeta(asset_id="a1"),
                    ),
                ],
            ),
            TrackSpec(name=OVERLAY_TRACK, items=[GapSpec(dur_s=25.0)]),
        ],
    )
)

engine = make_engine("sqlite:///:memory:")
init_models(engine)
store = CheckpointStore(make_session_factory(engine))

toolbox = DirectorToolbox(timeline, PROJECT, resolution=(1280, 720))
registry = build_director_registry(toolbox)
backend = build_backend(registry)
print(f"backend: {backend.provider} model={backend.model} base={backend.base_url}")

state = (
    "project=gemini_smoke\n"
    "master track (25.0s total) — item indexes:\n"
    "  [0] RAW src=0.00s dur=12.00s blur=0 zoom=0 audio=loudnorm_48k\n"
    "  [1] RAW src=12.00s dur=8.00s blur=0 zoom=0 audio=loudnorm_48k\n"
    "  [2] RAW src=20.00s dur=5.00s blur=0 zoom=0 audio=loudnorm_48k\n\n"
    "PERCEPTION BRIEF\n"
    "raw window: 25.0s\n"
    'speech beats (1): 3.0-18.5s Officer: "Step out of the vehicle for me sir..." (he is '
    "visibly intoxicated, slurring, refuses the FST)\n"
    "audio events: 12.0-16.0s Music (car radio, peak 0.7)\n\n"
    "EDIT MISSION: drop the dead air, tighten, narrate one beat, add a location card. "
    "Respond ONLY with function calls."
)
steps = backend.propose(state, "director")
print(f"proposed {len(steps)} step(s):")
for st in steps:
    print(f"  {st.tool} {st.args}")
usage = backend.last_usage
print(f"usage: {usage}")
print(f"system prompt bytes: {len(backend.last_system_prompt)}")
assert usage and usage.get("tokens_in", 0) > 0, "no token usage — request did not reach Gemini"
print("GEMINI LIVE: OK")
