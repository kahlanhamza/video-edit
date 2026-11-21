#!/usr/bin/env python3
"""LLM bridge: z-ai CLI (sandbox-native) as primary provider, with JSON parsing.

llm_chat(prompt)  -> str   (text completion)
llm_vision(prompt, image_path) -> str
"""
import json, os, subprocess, tempfile

def _run_cli(args, timeout=300):
    out = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    out.close()
    try:
        r = subprocess.run(args + ["-o", out.name], capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"z-ai CLI failed rc={r.returncode}: {r.stderr[-300:]}")
        with open(out.name, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and d.get("choices"):
            return d["choices"][0]["message"]["content"]
        return json.dumps(d)
    finally:
        try:
            os.unlink(out.name)
        except OSError:
            pass

def llm_chat(prompt, timeout=300):
    return _run_cli(["z-ai", "chat", "-p", prompt], timeout=timeout)

def llm_vision(prompt, image_path, timeout=300):
    return _run_cli(["z-ai", "vision", "-p", prompt, "-i", image_path], timeout=timeout)

if __name__ == "__main__":
    print(llm_chat("Reply with exactly: BRIDGE_OK")[:80])
