# SKILL.md — director-os Development Rules (the operating system)

**This file is binding.** Every session — human or agent — works exactly like this.
It encodes the issue-first, evidence-first loop proven on an 11-day, 267-commit
shipping project and adapted to this monorepo. When a rule here conflicts with a
chat summary or a memory, **this file wins**.

---

## 0. Session bootstrap (before ANY work — no exceptions)

You wake up with no memory. Do not pretend otherwise. In order:

1. **Restore identity**: `git status --branch`, `git log --oneline -10`, `git push` dry-run rights. Verify — never assume.
2. **Restore secrets** (`.env` is gitignored; the environment wipes it on recycle). Required keys: `VISION_API_KEY` (Gemini), `GLADIA_API_KEY`, `KAGGLE_USERNAME`+`KAGGLE_KEY`, `ELEVENLABS_API_KEY`+`ELEVENLABS_VOICE_ID`, `MODAL_TOKEN_ID`+`MODAL_TOKEN_SECRET` (dormant), `GH_TOKEN` for CI/issue triage.
3. **Read the record**: last rows of `STATUS.md`. That IS your memory. Never start from a chat summary alone.
4. **Triage CI**: `gh run list` / API — a red main is a stop-the-line event. Fix gates BEFORE new features (this bit us: two commits sat CI-RED on a format gate).
5. **State the plan** as a numbered TODO derived from STATUS.md + the user's request, then execute continuously — no permission-asking pauses.

## 1. Wire facts (the anti-amnesia reflex)

Before designing, implementing, or claiming anything, read the actual thing:
actual source file, actual CI job log (not the red X — read job verdicts and
annotations), actual API response, actual bytes on disk. In this project this
reflex caught: CI "failures" that were a broken branch filter (`ain]`), a format
gate that 12 files silently missed, and mypy errors hiding behind the format
failure. Distinguish **code-RED** (your change broke it) from **infra-RED**
(jobs dying at start — billing, quota, runners) before writing any fix.

## 2. Issue-first (nothing untracked)

Every lane starts as a GitHub issue with a design note and acceptance criteria.
Milestones M2–M5 are the rungs (#12 = M4 vision, #11 = M3 studio, #13 = M5
cloud); a milestone closes when its rungs close. Re-sequencing and pivots are
recorded on the issue, never decided silently.

## 3. RED before GREEN

Prove the gap before filling it: a failing gate run, a reproduced bug, a leaked
frame in a render. Post RED evidence on the issue; then fix; then post GREEN
evidence (commands + exit codes + test names). An issue closed without evidence
is a rumor.

## 4. Gates are non-bypassable

Local gates on the **committed tree** before every push (see `Makefile`):

```
uv run ruff check .          # lint
uv run ruff format --check . # format — CI canonicalized to ruff format
uv run mypy packages apps    # types
uv run pytest                # full suite (every bug becomes a test — PLAN §9)
```

A red gate is a stop, never a "skip it this time". Every guardrail lives as an
executable test (`tests/unit/test_g*.py`) — the 8 PLAN §9 traps are the precedent.

## 5. One lane = one commit; always push

Commit messages tell the story (`fix(gates): CI RED triaged — ...`). Push after
**every** lane — the sandbox recycles and context windows compact; only pushed
work survives. After push, chase the CI run to a verdict (API poll); never leave
a run un-triaged. Then append the STATUS.md row and push that too.

## 6. Sandbox reality (this environment)

- Long runs are **foreground `timeout` + resume-safe** design (segment cache is
  kill-safe; sweeps are idempotent). Background processes die between tool calls.
- Never trust the model file to exist: `yolov8n-face` degrades to packaged YuNet.
- Bash output goes through `> /tmp/x.txt 2>&1; echo exit=$?; tail` — garbled
  tool output is a trap, not a mystery.
- pydantic v2: `model_copy(update={...})` skips validation — construct real
  models (`BBox(...)`) when mutating geometry.

## 7. Secrets discipline

`.env` is local-only, gitignored AND in `.git/info/exclude`. Never print tokens;
never paste them in commits or logs. A key that appeared in chat once is rotated
on the next recycle. Payment-method decisions are recorded, not improvised:
Modal stays dormant behind `MODAL_TOKEN_ID` (card-verification fees are
deliberately avoided); free compute = LocalOnnxCpu (default) + Kaggle Kernels
(`KGAT` token, bearer auth); paid narration = ElevenLabs via one `.env` flip.

## 8. Architecture doctrine (do not fork it)

- **Ports, not imports**: LLMBackend / TtsBackend / ComputeBackend /
  VisionClient — adapters duck-type a Protocol, selection via env switch, zero
  heavy SDKs (pure httpx/onnxruntime).
- **Verify-before-declare**: never claim a render/release works; ffprobe/QA it.
- **Engine first**: packages/* before apps/* chrome; the renderer and QA are the
  product, the studio is a window onto them.
- **Every failure soft**: a backend that can't run skips its stage loudly, never
  breaks the pipeline.

## 9. The record

`STATUS.md` is the single source of truth, appended every task, pushed every
task. Dense rows: date, lane, what actually happened (including surprises),
outcome, commits. When a rule bites, the lesson goes in the row — war stories
are born there.
