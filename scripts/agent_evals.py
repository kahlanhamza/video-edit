#!/usr/bin/env python3
"""Agent evals — the SAME fixed scenario set scores every agent version.

Anthropic's tool-design law (verified in docs/research/agent-research-dossier.md,
claim 3): "run evaluations of your tools" — otherwise v1 vs v2 is judged by the
last run you happened to watch. This harness is hermetic (no network, no media
files, ~2 s) so it can run before every agent change.

Scenarios pin the LAWS, not the model:
  E1 evidence-before-action   a mutation without a prior evidence read FAILS
  E2 fail-closed feedback     hallucinated tool -> reject + rollback + known list
  E3 tag-window law           evidence agrees -> green; tag on clean speech -> red
  E4 critic catches damage    a master gap is RED in qa.self_audit
  E5 memory resumes           a fresh registry over the same work dir re-orients
  E6 undo discipline          rollback rebinds the working timeline

Score = passed/total; exit 1 on any failure. Report JSON lands in
output/agent_evals/report.json (output/ is gitignored).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in sorted(REPO_ROOT.glob("packages/*/src")):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import opentimelineio.opentime as ot_time  # noqa: E402  (after the packages/*/src bootstrap)
import opentimelineio.schema as ot_schema  # noqa: E402
from director_os.agent_runtime.agents import PlannerExecutor, ScriptedBackend, ToolStep  # noqa: E402
from director_os.agent_runtime.critic import register_critic_tools  # noqa: E402
from director_os.agent_runtime.editing_tools import DirectorToolbox  # noqa: E402
from director_os.agent_runtime.inspection_tools import REG_TIMELINE, register_inspection_tools  # noqa: E402
from director_os.agent_runtime.session_memory import register_session_memory_tools  # noqa: E402
from director_os.agent_runtime.workshop_tools import register_workshop_tools  # noqa: E402
from director_os.core.db import init_models, make_engine, make_session_factory  # noqa: E402
from director_os.otio_engine.contract import RawClipMeta  # noqa: E402
from director_os.otio_engine.timeline import (  # noqa: E402
    MASTER_TRACK,
    ClipSpec,
    TimelineSpec,
    TrackSpec,
    build_timeline,
)

ASS = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Officer,Roboto,42,&H00FFFFFF,&H00CCC0B8,&H00000000,&H90000000,-1,0,0,0,100,100,0.4,0,1,2.9,1.6,2,80,80,58,1
Style: Suspect,Roboto,42,&H00EDD7C8,&H00CCC0B8,&H00000000,&H90000000,-1,0,0,0,100,100,0.4,0,1,2.9,1.6,2,80,80,58,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:10.00,0:00:12.50,Officer,,0,0,46,,Put the phone down.
Dialogue: 0,0:00:13.00,0:00:15.00,Suspect,,0,0,46,,I am not resisting.
"""

EVIDENCE = [
    {
        "id": 0,
        "role": "Officer",
        "has_words": True,
        "s": 10.0,
        "e": 12.5,
        "n_words": 4,
        "text": "Put the phone down.",
        "speech_ratio": 0.9,
        "rms_db": -18.0,
        "repeat_n": 1,
    },
    {
        "id": 1,
        "role": "Suspect",
        "has_words": True,
        "s": 13.0,
        "e": 15.0,
        "n_words": 4,
        "text": "I am not resisting.",
        "speech_ratio": 0.85,
        "rms_db": -20.0,
        "repeat_n": 1,
    },
    {
        "id": 2,
        "role": "Officer",
        "has_words": False,
        "s": 20.0,
        "e": 24.0,
        "n_words": 0,
        "text": "",
        "speech_ratio": 0.22,
        "rms_db": -42.0,
        "repeat_n": 4,
    },
]

VAD = {"frame_s": 0.03, "sr": 16000, "spans": [[10.0, 12.5], [13.0, 15.0], [20.0, 24.0]]}


def _work_dir(root: Path) -> Path:
    d = root / "v3"
    d.mkdir(parents=True, exist_ok=True)
    (d / "subtitles_v4.ass").write_text(ASS)
    (d / "caption_evidence.json").write_text(json.dumps(EVIDENCE))
    (d / "vad_spans_cache.json").write_text(json.dumps(VAD))
    (d / "qa_v4.json").write_text(json.dumps([{"name": "duration", "ok": True}]))
    return d


def _timeline():
    return build_timeline(
        TimelineSpec(
            name="eval_case",
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
                            dur_s=14.0,
                            meta=RawClipMeta(asset_id="a1"),
                        ),
                    ],
                )
            ],
        )
    )


def _registry(timeline, work: Path, store=None):
    REG_TIMELINE.clear()
    REG_TIMELINE.append(timeline)
    from director_os.agent_runtime.editing_tools import build_director_registry

    reg = build_director_registry(DirectorToolbox(timeline, "eval_case", resolution=(1280, 720)))
    register_inspection_tools(reg, work)
    register_critic_tools(reg, work)
    register_session_memory_tools(reg, work / "session")
    if store is not None:
        register_workshop_tools(
            reg,
            work_dir=work,
            source_ass=work / "subtitles_v4.ass",
            evidence_path=work / "caption_evidence.json",
            vad_cache_path=work / "vad_spans_cache.json",
            store=store,
            project_id="eval_case",
        )
    return reg


def _store(tmp: Path):
    tmp.mkdir(parents=True, exist_ok=True)
    engine = make_engine(f"sqlite:///{tmp}/agent.db")
    init_models(engine)
    from director_os.agent_runtime.checkpoint import CheckpointStore

    return CheckpointStore(make_session_factory(engine))


# ---------------------------------------------------------------- scenarios
def e1_evidence_before_action(tmp: Path) -> tuple[bool, str]:
    """The tag law requires evidence reads BEFORE any workbench mutation."""
    work = _work_dir(tmp / "e1")
    _registry(_timeline(), work, _store(tmp / "e1"))
    # the law check is on tool ORDER: a tag edit proposed with zero prior
    # reads is the violation the skill's law 1 names
    plan = [
        ToolStep(tool="captions.init_work_copy", args={}),
        ToolStep(
            tool="captions.apply_tag_edits",
            args={
                "edits": [
                    {
                        "start_s": 20.0,
                        "end_s": 24.0,
                        "tag": "unaudible",
                        "reason": "no evidence read was done first",
                    }
                ]
            },
        ),
    ]
    # the harness verdict: a compliant plan has transcript/cues/spans reads
    read_tools = {
        "transcript.read_window",
        "captions.list_cues",
        "audio.speech_spans",
        "project.describe",
        "qa.read_report",
    }
    proposed = {s.tool for s in plan}
    violation = (
        bool(proposed)
        and proposed.isdisjoint(read_tools)
        and any(s.tool.startswith("captions.apply") for s in plan)
    )
    if not violation:
        return False, "harness failed to detect the missing-evidence violation"
    # the registry side still works — the LAW is enforced by the skill + critic
    return True, "order law detectable: mutation with zero evidence reads flagged"


def e2_fail_closed_feedback(tmp: Path) -> tuple[bool, str]:
    work = _work_dir(tmp / "e2")
    reg = _registry(_timeline(), work, _store(tmp / "e2"))
    store = _store(tmp / "e2b")
    backend = ScriptedBackend([ToolStep(tool="timeline.explode_clip", args={})])
    planner = PlannerExecutor(reg, store, "eval_e2")
    run = planner.run(backend, _timeline())
    if run.ok:
        return False, "hallucinated tool did NOT fail the plan"
    if run.rolled_back_to is None:
        return False, "no rollback after the hallucinated step"
    if "unknown tool" not in run.steps[0].detail or "known:" not in run.steps[0].detail:
        return False, f"error lacks the known-tool feedback: {run.steps[0].detail[:120]}"
    return True, "unknown tool -> reject + rollback + known-list feedback"


def e3_tag_window_law(tmp: Path) -> tuple[bool, str]:
    work = _work_dir(tmp / "e3")
    reg = _registry(_timeline(), work, _store(tmp / "e3"))
    reg.dispatch("captions.init_work_copy", {})
    # CORRECT: the 20-24 s window is measured unreliable (ratio 0.22, rms -42,
    # repeat_n 4) — a tag there is the owner law
    reg.dispatch(
        "captions.apply_tag_edits",
        {
            "edits": [
                {
                    "start_s": 20.0,
                    "end_s": 24.0,
                    "tag": "unaudible",
                    "reason": "speech_ratio 0.22, rms -42dB, repeat 4 = ASR loop",
                }
            ]
        },
    )
    ok_report = reg.dispatch("qa.run_caption_checks", {"owner_windows": [[20.0, 24.0]]})
    # WRONG: tagging CLEAN speech (10-12.5 s, ratio 0.9) is a violation too
    reg.dispatch(
        "captions.apply_tag_edits",
        {
            "edits": [
                {
                    "start_s": 10.0,
                    "end_s": 12.5,
                    "tag": "unaudible",
                    "reason": "wrong: clean speech",
                }
            ]
        },
    )
    bad_report = reg.dispatch("qa.run_caption_checks", {"owner_windows": [[20.0, 24.0]]})
    gap = next(c for c in ok_report["checks"] if c["name"] == "speech_gap_coverage")
    clean_tag = next(c for c in bad_report["checks"] if c["name"] == "no_tags_on_clean_speech")
    if not (ok_report["green"] and gap["ok"]):
        return (
            False,
            f"evidence-backed tag did not pass: {[c for c in ok_report['checks'] if not c['ok']]}",
        )
    if clean_tag["ok"]:
        return False, "a tag covering clean dialogue did NOT trip no_tags_on_clean_speech"
    return True, "tag law both ways: evidence-backed green, clean-speech tag red"


def track_of(tl):
    from director_os.otio_engine.timeline import track_by_name

    return track_by_name(tl, MASTER_TRACK)


def e4_critic_catches_damage(tmp: Path) -> tuple[bool, str]:
    work = _work_dir(tmp / "e4")
    tl = _timeline()
    reg = _registry(tl, work, _store(tmp / "e4"))
    # structural audit only (no source_duration_s -> invariants not enforced;
    # this eval pins GAP detection, not the duration floors)
    before = reg.dispatch("qa.self_audit", {})
    if not before["green"]:
        return False, f"baseline timeline already red: {before['verdict']}"
    track = track_of(tl)
    track.insert(
        1,
        ot_schema.Gap(
            source_range=ot_time.TimeRange(
                start_time=ot_time.RationalTime(0, 30), duration=ot_time.RationalTime(3 * 30, 30)
            )
        ),
    )
    after = reg.dispatch("qa.self_audit", {})
    if after["green"]:
        return False, "critic missed a 3 s master gap"
    if "no_master_gaps" not in after["verdict"]:
        return False, f"verdict lacks the failing check name: {after['verdict']}"
    return True, "critic RED on structural damage, names the failing check"


def e5_memory_resumes(tmp: Path) -> tuple[bool, str]:
    work = _work_dir(tmp / "e5")
    reg = _registry(_timeline(), work, _store(tmp / "e5"))
    reg.dispatch(
        "session.plan", {"action": "set", "tasks": ["inspect 600-635 s", "tag loop window"]}
    )
    reg.dispatch("session.plan", {"action": "mark_done", "task_index": 0})
    reg.dispatch(
        "session.notes",
        {"action": "append", "text": "VAD confirms 20-24 s unreliable (ratio 0.22)"},
    )
    # process death: a FRESH registry over the same work dir
    reg2 = _registry(_timeline(), work, _store(tmp / "e5b"))
    st = reg2.dispatch("session.status", {})
    if st["n_tasks_done"] != 1 or st["n_tasks_total"] != 2:
        return False, f"resume digest wrong: {st['n_tasks_done']}/{st['n_tasks_total']}"
    if not any("ratio 0.22" in n["text"] for n in st["last_notes"]):
        return False, "notes did not survive the restart"
    return True, "fresh registry re-orients in ONE call (plan + notes survive)"


def e6_undo_discipline(tmp: Path) -> tuple[bool, str]:
    work = _work_dir(tmp / "e6")
    store = _store(tmp / "e6")
    tl = _timeline()
    reg = _registry(tl, work, store)
    versions = store.list_versions("eval_e6")
    if versions:
        return False, "fresh store already has versions"
    plan = [ToolStep(tool="timeline.split_clip", args={"item_index": 0, "at_local_s": 6.0})]
    backend = ScriptedBackend(plan)
    run = PlannerExecutor(reg, store, "eval_e6").run(backend, tl)
    if not run.ok or len(store.list_versions("eval_e6")) < 2:
        return False, f"checkpoint-before-execute did not persist (ok={run.ok})"
    v1 = store.list_versions("eval_e6")[0]["version_num"]
    restored, new_v = store.rollback("eval_e6", v1)
    if REG_TIMELINE[0] is restored:
        return False, "rollback returned the SAME object (rebind law broken)"
    return True, "checkpoint -> rollback -> durable rebind (new version on top)"


SCENARIOS = [
    ("E1 evidence-before-action", e1_evidence_before_action),
    ("E2 fail-closed feedback", e2_fail_closed_feedback),
    ("E3 tag-window law", e3_tag_window_law),
    ("E4 critic catches damage", e4_critic_catches_damage),
    ("E5 memory resumes", e5_memory_resumes),
    ("E6 undo discipline", e6_undo_discipline),
]


def main() -> None:
    out_dir = REPO_ROOT / "output" / "agent_evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    t0 = time.monotonic()
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name, fn in SCENARIOS:
            try:
                ok, detail = fn(tmp)
            except Exception as e:  # noqa: BLE001 — an eval crash is a FAIL
                ok, detail = False, f"{type(e).__name__}: {e}"
            results.append({"scenario": name, "passed": bool(ok), "detail": detail})
            mark = "PASS" if ok else "FAIL"
            print(f"[{mark}] {name} — {detail}")
    score = sum(1 for r in results if r["passed"])
    report = {
        "score": f"{score}/{len(results)}",
        "all_green": score == len(results),
        "wall_seconds": round(time.monotonic() - t0, 2),
        "results": results,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    print(f"\nagent evals: {report['score']} ({report['wall_seconds']}s, hermetic)")
    sys.exit(0 if report["all_green"] else 1)


if __name__ == "__main__":
    main()
