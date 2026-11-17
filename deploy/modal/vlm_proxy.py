"""Tiny Gemini egress proxy on Modal — unblocks geo-restricted worker regions.

Some datacenter regions get 400 FAILED_PRECONDITION ("User location is not
supported") from the Generative Language API. Modal's US/EU egress does not.
This app is a ~30MB image that builds in seconds and hosts one function:

    vlm_call(model, payload_json) -> {"status": int, "body": str}

The CPU-side `perception.vision.GeminiVisionClient` detects the geo-block on
its direct call and transparently retries through here (decision sticks for
the client's lifetime).

Deploy (VISION_API_KEY rides a Modal Secret baked at deploy time):
    set -a; . ./.env; set +a; modal deploy deploy/modal/vlm_proxy.py
"""

from __future__ import annotations

import os

import modal

app = modal.App("directoros-vlm")

vlm_image = modal.Image.debian_slim(python_version="3.12").pip_install("requests")

vlm_secret = modal.Secret.from_local_environ(["VISION_API_KEY"])


@app.function(image=vlm_image, secrets=[vlm_secret], timeout=120)
def vlm_call(model: str, payload_json: str, api_version: str = "v1beta") -> dict:
    import requests

    r = requests.post(
        f"https://generativelanguage.googleapis.com/{api_version}/models/{model}:generateContent",
        headers={"x-goog-api-key": os.environ["VISION_API_KEY"]},
        data=payload_json,
        timeout=90,
    )
    return {"status": r.status_code, "body": r.text[:100_000]}


@app.local_entrypoint()
def main(model: str = "gemini-3.5-flash-lite"):
    import json

    out = vlm_call.remote(
        model=model,
        payload_json=json.dumps(
            {"contents": [{"role": "user", "parts": [{"text": "Reply with the single word: ok"}]}]}
        ),
    )
    print(json.dumps(out, indent=2)[:600])
