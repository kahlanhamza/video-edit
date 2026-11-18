"""Live routing smoke: Stage 6 Director tool-calling + Stage 4 Vision, real keys.

Loads .env (gitignored), then:
- Stage 6: build_backend_chain(mini director registry).propose() — the Groq
  tier should serve; if it 429s the chain walks OpenRouter -> Gemini.
- Stage 4: build_vision_client().analyze() on one real frame — Gemini
  flash-lite primary; OpenRouter :free fallback on capacity.

Exit code 0 = both stages 200 OK end-to-end.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    import os

    env_path = ROOT / ".env"
    if not env_path.exists():
        print("no .env — aborting (never smoke-test with guessed keys)")
        sys.exit(2)
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())
    os.environ["OUTPUT_DIR"] = str(ROOT / "output")
    from director_os.core.config import get_settings

    get_settings.cache_clear()


def stage6_director() -> tuple[bool, str]:
    from director_os.agent_runtime.llm import build_backend_chain
    from director_os.agent_runtime.registry import ToolRegistry

    reg = ToolRegistry()

    @reg.tool("timeline.trim_clip", description="Trim a clip to a new start/duration window.")
    def trim_clip(item_index: int, new_start_s: float, new_dur_s: float) -> str:
        return f"trimmed item {item_index}"

    @reg.tool("timeline.remove_clip", description="Remove a clip from the master track.")
    def remove_clip(item_index: int) -> str:
        return f"removed item {item_index}"

    backend = build_backend_chain(reg)
    tiers = (
        backend.tier_names
        if hasattr(backend, "tier_names")
        else [f"{getattr(backend, 'provider', '?')}:{getattr(backend, 'model', '')}"]
    )
    print(f"  tier chain: {tiers}")
    state = (
        "project=eustis_smoke master_items=[Clip(raw_1 0.0-12.0s), Clip(raw_2 12.0-25.0s), "
        "Clip(raw_3 25.0-31.0s)] — propose 1-2 tool calls."
    )
    steps = backend.propose(state, "director")
    usage = getattr(backend, "last_usage", None)
    active = getattr(backend, "active", backend)
    print(f"  served by: {getattr(active, 'provider', '?')}:{getattr(active, 'model', '')}")
    print(f"  usage: {usage}")
    print(f"  steps: {[(s.tool, s.args) for s in steps]}")
    record = getattr(backend, "record_results", None)
    if callable(record) and steps:
        record(steps, ["smoke-ok"] * len(steps))
        print("  record_results round-trip: OK")
    if not steps:
        return False, "no tool steps proposed"
    return True, f"served by {getattr(active, 'provider', '?')}:{getattr(active, 'model', '')}"


def stage4_vision() -> tuple[bool, str]:
    from director_os.perception.vision import FrameRef, build_vision_client

    frame_png = Path("/tmp/routing_smoke_frame.jpg")
    src = ROOT / "tests" / "fixtures" / "fixture_30s.mp4"
    if not frame_png.exists():
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                "5",
                "-i",
                str(src),
                "-frames:v",
                "1",
                "-q:v",
                "4",
                str(frame_png),
            ],
            check=True,
        )
    client = build_vision_client()
    print(
        f"  client chain: {getattr(client, 'provider', type(client).__name__)}"
        + (
            f" tiers={[type(c).__name__ for c in client._clients]}"
            if hasattr(client, "_clients")
            else ""
        )
    )
    frames = [FrameRef(timestamp_s=5.0, image_path=frame_png)]
    insights = client.analyze("Describe what is visible in this frame in one observation.", frames)
    active = getattr(client, "active", client)
    print(f"  served by: {getattr(active, 'provider', '?')}:{getattr(active, 'model', '')}")
    print(f"  insights: {json.dumps([i.model_dump() for i in insights])[:400]}")
    if not insights:
        return False, "no insights returned"
    return True, f"{len(insights)} insight(s) via {getattr(active, 'provider', '?')}"


def main() -> None:
    load_env()
    ok6, msg6 = stage6_director()
    print(f"STAGE 6 DIRECTOR: {'PASS' if ok6 else 'FAIL'} — {msg6}\n")
    ok4, msg4 = stage4_vision()
    print(f"STAGE 4 VISION: {'PASS' if ok4 else 'FAIL'} — {msg4}")
    sys.exit(0 if (ok6 and ok4) else 1)


if __name__ == "__main__":
    main()
