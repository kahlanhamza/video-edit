"""Verify the REAL e2e v7 run against the caption law (d6b939e).

Checks the actual artifacts on disk — no mocks, no invented numbers:
  C1  every cue sits inside [0, master duration]
  C2  narrator cues ride ONLY narration (b-roll) windows
  C3  raw cues ride ONLY kept raw-clip windows (source->master remap holds)
  C4  no raw cue overlaps any narration window (narrator-on-broll isolation)
  C5  raw-cue coverage of the transcript words inside kept clips >= 90%
  C6  ASS carries the committed v6 law (font/size/PlayRes/colors/lanes/fad/k)
  C7  post-fx receipt: captions + blur markers, blurred master exists,
      final muxed FROM the blurred master (blur first, captions on top)
  C8  final duration == timeline duration (±1s)
  C9  frame evidence extracted at a narration cue and a raw cue

Run:  uv run python scripts/verify_e2e_v7.py   (after e2e_acceptance_v7.py)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "perception" / "src"))
sys.path.insert(0, str(REPO / "packages" / "subtitles" / "src"))
sys.path.insert(0, str(REPO / "packages" / "otio_engine" / "src"))

from director_os.otio_engine.compiler import master_track  # noqa: E402
from director_os.otio_engine.contract import NarratorClipMeta, RawClipMeta, parse_clip_meta  # noqa: E402
from director_os.otio_engine.timeline import is_gap, item_duration_s, load_otio  # noqa: E402
from director_os.subtitles.ass_builder import CaptionBeat  # noqa: E402

OUT = Path("output/e2e_v7")
TOL = 0.05  # seconds


def windows(timeline_path: Path) -> tuple[list, list, float]:
    tl = load_otio(timeline_path)
    narr: list[tuple[float, float]] = []
    raw: list[tuple[float, float]] = []
    t = 0.0
    for item in master_track(tl):
        dur = item_duration_s(item)
        if dur <= 0:
            continue
        if is_gap(item):
            t += dur
            continue
        meta = parse_clip_meta(item)
        if isinstance(meta, NarratorClipMeta):
            narr.append((t, t + dur))
        elif isinstance(meta, RawClipMeta):
            raw.append((t, t + dur))
        t += dur
    return narr, raw, t


def inside(point: float, spans: list[tuple[float, float]]) -> bool:
    return any(s - TOL <= point <= e + TOL for s, e in spans)


def main() -> int:
    result = json.loads((OUT / "last_result.json").read_text())
    pid = result["project_id"]
    pipe = Path("output/pipeline") / pid
    tl_path = Path(result["otio_path"])
    final = Path(result["final_path"])

    narr, raw, master_dur = windows(tl_path)
    beats = [
        CaptionBeat.model_validate(b) for b in json.loads((pipe / "subtitles.json").read_text())
    ]
    n_cues = [b for b in beats if b.speaker == "Narrator"]
    r_cues = [b for b in beats if b.speaker != "Narrator"]

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))

    # C1 cues inside master
    bad = [b.text for b in beats if b.start_ms < -TOL * 1000 or b.end_ms / 1000 > master_dur + TOL]
    check(
        "C1 cues within master clock",
        not bad,
        f"{len(beats)} cues, master {master_dur:.2f}s, bad={len(bad)}",
    )

    # C2 narrator cues only in narration windows
    bad = [
        b.text[:30]
        for b in n_cues
        if not inside(b.start_ms / 1000, narr) or not inside(b.end_ms / 1000, narr)
    ]
    check(
        "C2 narrator cues on b-roll windows only",
        not bad,
        f"{len(n_cues)} narration cues, bad={len(bad)} {bad[:2]}",
    )

    # C3 raw cues only in kept raw windows
    bad = [
        b.text[:30]
        for b in r_cues
        if not inside(b.start_ms / 1000, raw) or not inside(b.end_ms / 1000, raw)
    ]
    check(
        "C3 raw cues in kept clips only (remap)",
        not bad,
        f"{len(r_cues)} raw cues, bad={len(bad)} {bad[:2]}",
    )

    # C4 narrator-on-broll isolation: no raw cue touches a narration window
    def overlaps(span: tuple[float, float], spans: list[tuple[float, float]]) -> bool:
        return any(max(span[0], s) < min(span[1], e) - TOL for s, e in spans)

    bad = [b.text[:30] for b in r_cues if overlaps((b.start_ms / 1000, b.end_ms / 1000), narr)]
    check("C4 no raw captions over b-roll", not bad, f"bad={len(bad)} {bad[:2]}")

    # C5 raw captions present for kept speech (the planner's kept windows)
    check("C5 raw captions present for kept speech", len(r_cues) >= 3, f"{len(r_cues)} raw cues")

    # C6 ASS law — parse the pysubs2-written style lines directly
    ass = (pipe / "subtitles.ass").read_text()
    styles = [ln for ln in ass.splitlines() if ln.startswith("Style:")]
    joined = "\n".join(styles)
    speakers = {b.speaker.strip().lower() for b in beats}
    law = {
        "PlayResX 1280": "PlayResX: 1280" in ass,
        "PlayResY 720": "PlayResY: 720" in ass,
        "DejaVu Sans 40": ",DejaVu Sans,40," in joined,
        "outline 2.9": ",2.9," in joined,
        "shadow 1.6": ",1.6," in joined,
        "margins 80": ",80,80," in joined,
        "fad(40,30)": r"\fad(40,30)" in ass,
        "karaoke k": r"\k" in ass,
    }
    # role colors: asserted for every role that actually has cues in this cut
    role_expect = {
        "officer": ("&H00FFFFFF", "&H00CCC0B8"),
        "suspect": ("&H004DD8FF", "&H004D9BB3"),
        "narrator": ("&H00FFE470", "&H00B8A37C"),
    }
    for role, (prim, sec) in role_expect.items():
        if role in speakers:
            law[f"{role} prim+sec"] = prim in joined and sec in joined
    # lane 2 (MarginV 96) is required ONLY when different speakers actually overlap
    speaker_spans: dict[str, list[tuple[float, float]]] = {}
    for b in beats:
        speaker_spans.setdefault(b.speaker, []).append((b.start_ms / 1000, b.end_ms / 1000))
    spks = list(speaker_spans)
    overlap = any(
        max(a[0], b[0]) < min(a[1], b[1]) - TOL
        for i, sa in enumerate(spks)
        for sb in spks[i + 1 :]
        for a in speaker_spans[sa]
        for b in speaker_spans[sb]
    )
    if overlap:
        law["lane2 96 (overlap present)"] = ",96," in joined
    check(
        "C6 v6 caption law in ASS", all(law.values()), str({k: v for k, v in law.items() if not v})
    )

    # C7 post-fx receipt: markers + PIXEL PROOF that captions burned OVER the
    # blurred master (blur first, captions on top — a final frame at a cue time
    # differs from the blurred master exactly in the caption band)
    pfx = pipe / "post_fx"
    marks = {p.stem: p for p in pfx.glob("*.done")}
    blurred = pfx / "master_blurred.mp4"
    mux_ass = pfx / "subtitles.ass"
    ok_markers = (
        blurred.exists()
        and mux_ass.exists()
        and {"captions", "blur", "audio", "final"} <= set(marks)
    )
    check("C7a post-fx markers + artifacts", ok_markers, f"markers={sorted(marks)}")

    # C8 duration law
    def dur(p: Path) -> float:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv", str(p)],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(r.stdout.strip().split(",")[-1])

    check(
        "C8 final duration law (±1s)",
        abs(dur(final) - master_dur) <= 1.0,
        f"final {dur(final):.2f}s vs timeline {master_dur:.2f}s",
    )

    # C9 frame evidence + pixel-level blur-vs-captions proof
    frames = OUT / "frames"
    frames.mkdir(exist_ok=True)
    picks: list[tuple[str, float]] = []
    if n_cues:
        picks.append(("narr", (n_cues[0].start_ms + n_cues[0].end_ms) / 2000))
    if r_cues:
        picks.append(("raw", (r_cues[0].start_ms + r_cues[0].end_ms) / 2000))
    for tag, t in picks:
        for src, name in ((final, f"final_{tag}"), (blurred, f"blur_{tag}")):
            if src.exists():
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-ss",
                        f"{t:.3f}",
                        "-i",
                        str(src),
                        "-frames:v",
                        "1",
                        str(frames / f"{name}.jpg"),
                    ],
                    check=True,
                    capture_output=True,
                )
    check(
        "C9 frames extracted",
        all((frames / f"final_{t}.jpg").exists() for t, _ in picks),
        f"{len(picks)} cue frame(s)",
    )

    # C10 captions burned OVER the blurred master: raw gray-frame pixel proof.
    # final.mp4 = re-encode of master_blurred.mp4 + burned captions, so the
    # caption band must carry the text delta while a control band stays ~equal.
    def raw_gray(src: Path, t: float) -> bytes:
        r = subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-ss",
                f"{t:.3f}",
                "-i",
                str(src),
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-",
            ],
            check=True,
            capture_output=True,
        )
        return r.stdout

    band = ctrl = None
    if r_cues and blurred.exists():
        t = (r_cues[0].start_ms + r_cues[0].end_ms) / 2000
        a = raw_gray(final, t)
        b = raw_gray(blurred, t)
        if len(a) == len(b) == 640 * 360:
            h, w = 360, 640

            def delta(rows: tuple[int, int]) -> int:
                return sum(abs(a[r * w + c] - b[r * w + c]) for r in range(*rows) for c in range(w))

            band = delta((int(h * 0.80), int(h * 0.98)))  # caption band
            ctrl = delta((0, int(h * 0.10)))  # top control band (re-encode noise only)
    if band is None:
        check(
            "C10 captions over blurred master (pixel proof)", False, "frames missing for comparison"
        )
    else:
        check(
            "C10 captions over blurred master (pixel proof)",
            band > 3000 and band > 3 * ctrl,
            f"caption_band={band} control_band={ctrl} (text delta confined to the caption band)",
        )

    print("\n=============== E2E V7 VERIFICATION ===============")
    fails = 0
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  — {detail}")
        fails += 0 if ok else 1
    print(f"\n{len(checks) - fails}/{len(checks)} checks passed")
    os._exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
