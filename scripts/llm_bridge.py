#!/usr/bin/env python3
"""LLM bridge: z-ai CLI (sandbox-native) as primary provider, with JSON parsing.

llm_chat(prompt)  -> str   (text completion)
llm_vision(prompt, image_path) -> str
"""

import contextlib
import json
import os
import subprocess
import tempfile


def _run_cli(args, timeout=300):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_name = tmp.name
    try:
        r = subprocess.run(args + ["-o", out_name], capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"z-ai CLI failed rc={r.returncode}: {r.stderr[-300:]}")
        with open(out_name, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and d.get("choices"):
            return d["choices"][0]["message"]["content"]
        return json.dumps(d)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(out_name)


def llm_chat(prompt, timeout=300):
    return _run_cli(["z-ai", "chat", "-p", prompt], timeout=timeout)


def llm_vision(prompt, image_path, timeout=300):
    return _run_cli(["z-ai", "vision", "-p", prompt, "-i", image_path], timeout=timeout)


if __name__ == "__main__":
    print(llm_chat("Reply with exactly: BRIDGE_OK")[:80])
