"""Modal serverless GPU render target (PLAN §8 M4) — wiring stub.

M4 moves heavy renders (1080p long-form, vision batch inference) to Modal's
GPU fleet. This module defines the app + volume layout and the render function
contract; it activates when MODAL_TOKEN_ID/SECRET exist (input pending).

Local-first guardrail stays: the same `RenderSpec` runs via `otio_engine.renderer`
on CPU today, so Modal is an accelerator, not a fork of the engine.

Run: `modal run deploy/modal/gpu_render.py::render_segment --spec-json '...'`
"""

from __future__ import annotations

import json

import modal

app = modal.App("directoros-gpu")

cache_volume = modal.Volume.from_name("directoros-segments", create_if_missing=True)

gpu_image = (
    modal.Image.debian_slim(python_version="3.12").pip_install(
        "director-os-core", "ffmpeg-static"
    )  # workspace deps wired in deploy
)


@app.function(image=gpu_image, gpu="T4", volumes={"/cache": cache_volume}, timeout=3600)
def render_segment(spec_json: str) -> dict:
    """Render one `VideoRenderSpec` segment on GPU; product lands in /cache."""
    spec = json.loads(spec_json)
    # M4 close-out: import otio_engine renderer against /cache and execute.
    # Kept as a typed stub so the fleet entrypoint exists before credentials do.
    return {
        "ok": False,
        "reason": "modal render wiring pending credentials",
        "spec": spec.get("name", "spec"),
    }


@app.local_entrypoint()
def main(spec_json: str):
    result = render_segment.remote(spec_json)
    print(result)
