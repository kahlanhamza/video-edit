"""Live API smoke — the Studio's exact flow, no mocks.

Proves the one-button contract end to end through the REAL HTTP surface:

  1. builds a 12s spoken fixture (edge-tts radio chatter + testsrc2),
     an HTML "news article" and a real PDF
  2. serves all three over plain HTTP (the agent must download them)
  3. boots the FastAPI gateway (same `make api` entrypoint)
  4. POST /projects (video link + resource URLs) -> POST /run
  5. polls /jobs to human_gate (or a parked error), then GET /research

Run:  uv run python scripts/live_api_smoke.py   (.env is parsed automatically)
"""

from __future__ import annotations

import contextlib
import http.server
import json
import os
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "perception" / "src"))

OUT = REPO / "output" / "live_smoke"
PORT = 8931


def load_env() -> None:
    """Parse .env without shell sourcing (placeholder lines must not explode)."""
    env_file = REPO / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if not value or "<<<" in value:
            continue
        os.environ.setdefault(key.strip(), value)


ARTICLE = """<!doctype html><html><head><title>Elm Street incident briefing</title>
<style>.nav{display:none}</style></head><body>
<nav>home | about | junk</nav>
<article><h1>Elm Street incident briefing</h1>
<p>Police responded to a code four call at Fifth and Main at 14:32.
The subject was detained without injuries and released pending charges.</p>
<p>The department released the bodycam footage the same week.</p></article>
</body></html>"""


def make_pdf(text: str) -> bytes:
    obj1 = b"<< /Type /Catalog /Pages 2 0 R >>"
    obj2 = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
    obj3 = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>"
    )
    stream = f"BT /F1 18 Tf 72 700 Td ({text}) Tj ET".encode()
    obj4 = b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
    obj5 = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    header = b"%PDF-1.4\n"
    body = b""
    offsets = []
    for i, o in enumerate([obj1, obj2, obj3, obj4, obj5], start=1):
        offsets.append(len(header) + len(body))
        body += str(i).encode() + b" 0 obj\n" + o + b"\nendobj\n"
    xref_pos = len(header) + len(body)
    xref = b"xref\n0 6\n0000000000 65535 f \n"
    for off in offsets:
        xref += f"{off:010d} 00000 n \n".encode()
    trailer = (
        b"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n" + str(xref_pos).encode() + b"\n%%EOF"
    )
    return header + body + xref + trailer


def build_fixture(dest: Path) -> None:
    """12s clip: testsrc2 video + two TTS radio utterances, 48k stereo AAC."""
    import edge_tts

    lines = [
        (0.5, "Unit one to dispatch, we have a code four in progress at Fifth and Main."),
        (6.0, "Dispatch copies, show unit one en route, ETA two minutes."),
    ]
    parts = []
    for i, (_, text) in enumerate(lines):
        p = OUT / f"line{i}.mp3"
        import asyncio

        asyncio.run(edge_tts.Communicate(text, "en-US-BrianMultilingualNeural").save(str(p)))
        parts.append(p)
    delays = [f"adelay={int(d * 1000)}:all=1" for d, _ in lines]
    inputs = ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo:d=12"]
    for p in parts:
        inputs += ["-i", str(p)]
    filter_parts = [f"[{i + 2}:a]{d}[a{i}]" for i, d in enumerate(delays)]
    mix = "".join(f"[a{i}]" for i in range(len(parts)))
    filter_complex = ";".join(
        [*filter_parts, f"[1:a]{mix}amix=inputs={len(parts) + 1}:duration=first:normalize=0[out]"]
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=30:duration=12",
            *inputs,
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v",
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "30",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-t",
            "12",
            "-shortest",
            str(dest),
        ],
        check=True,
        capture_output=True,
    )


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(OUT), **kwargs)

    def log_message(self, *args):  # silence the per-request noise
        pass


def main() -> int:
    load_env()
    OUT.mkdir(parents=True, exist_ok=True)
    print("[smoke] building spoken fixture (edge-tts + ffmpeg)...")
    fixture = OUT / "fixture_speech.mp4"
    build_fixture(fixture)
    (OUT / "briefing.html").write_text(ARTICLE)
    (OUT / "evidence_memo.pdf").write_bytes(
        make_pdf("Evidence memo: code four call logged at 14:32, no injuries reported.")
    )

    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        base = f"http://127.0.0.1:{PORT}"

        # boot the API exactly like `make api`, on a fresh sqlite DB
        db_path = OUT / "smoke.db"
        env = dict(os.environ)
        env["DATABASE_URL"] = f"sqlite:///{db_path}"
        env["OUTPUT_DIR"] = str(OUT / "artifacts")
        api = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "director_os.api.main:app", "--port", "8902"],
            cwd=str(REPO),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            import urllib.request

            def api_req(method: str, path: str, body: dict | None = None):
                req = urllib.request.Request(
                    f"http://127.0.0.1:8902{path}",
                    method=method,
                    data=json.dumps(body).encode() if body is not None else None,
                    headers={"Content-Type": "application/json"},
                )
                try:
                    with urllib.request.urlopen(req, timeout=30) as r:
                        return r.status, json.loads(r.read() or b"{}")
                except urllib.error.HTTPError as e:
                    return e.code, json.loads(e.read() or b"{}")

            for _ in range(60):
                try:
                    if api_req("GET", "/health")[0] == 200:
                        break
                except Exception:
                    time.sleep(0.5)
            else:
                print(api.stdout.read().decode()[-2000:])
                return 1

            print("[smoke] API up — creating project with video link + resources")
            status, project = api_req(
                "POST",
                "/projects",
                {
                    "title": "Live API smoke",
                    "video_urls": [f"{base}/fixture_speech.mp4"],
                    "auto_discover": True,
                    "sources": [
                        {"kind": "url", "label": "briefing", "url": f"{base}/briefing.html"},
                        {"kind": "url", "label": "memo", "url": f"{base}/evidence_memo.pdf"},
                    ],
                },
            )
            assert status == 201, project
            pid = project["id"]
            print(f"[smoke] project {pid[:8]} — POST /run")

            run_body = {"video_url": f"{base}/fixture_speech.mp4"}
            status, run = api_req("POST", f"/projects/{pid}/run", run_body)
            assert status in (200, 202), run
            print(f"[smoke] run: reused={run['reused']} ({run['detail']})")

            # poll like the Studio SSE reader would (up to 8 minutes)
            deadline = time.time() + 480
            last = {}
            while time.time() < deadline:
                _, jobs = api_req("GET", f"/projects/{pid}/jobs")
                last = jobs[-1] if jobs else {}
                err = last.get("error")
                stage, pct = last.get("stage"), last.get("progress_pct", 0)
                tail = ("ERR: " + err[:120]) if err else ""
                print(f"  [{time.strftime('%H:%M:%S')}] {stage} {pct:.0f}% {tail}")
                if err or stage == "human_gate":
                    break
                time.sleep(4)

            _, sources = api_req("GET", f"/projects/{pid}/sources")
            srcs = [
                {"kind": x.get("kind"), "status": x.get("status"), "label": x.get("label")}
                for x in (sources if isinstance(sources, list) else sources.get("sources", []))
            ]
            print("[smoke] case sources:", json.dumps(srcs)[:600])
            status, dossier = api_req("GET", f"/projects/{pid}/dossier")
            if status == 200:
                d = dossier if isinstance(dossier, dict) else {}
                print("[smoke] dossier keys:", list(d)[:10])
                print("[smoke] dossier:", json.dumps(d)[:1200])

            _, assets = api_req("GET", f"/projects/{pid}/assets")
            print("[smoke] assets:", json.dumps(assets)[:400])

            ok = last.get("stage") == "human_gate" and not last.get("error")
            verdict = "PIPELINE GREEN (awaiting human review)" if ok else "PARKED"
            print(f"[smoke] RESULT: {verdict} {str(last.get('error'))[:200]}")
            result = 0 if ok else 2
        finally:
            api.terminate()
            with contextlib.suppress(Exception):
                api.wait(timeout=5)
            if api.poll() is None:
                api.kill()
                api.wait()
            sys.stdout.flush()
            os._exit(result)  # every helper thread must release the shell


if __name__ == "__main__":
    raise SystemExit(main())
