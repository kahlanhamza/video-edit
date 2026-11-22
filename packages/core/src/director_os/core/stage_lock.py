"""Content-hash stage-lock engine — "fix one stage, never rerun the world".

The product pipeline (`apps/worker/pipeline.py`) persists each stage's product
in a DOMAIN table, which is what makes crash-resume work inside DirectorOS.
This module generalizes the same guarantee for FILE-ARTIFACT pipelines (ops
drills, media production chains, local re-renders) and is the engine behind
`tests/unit/test_stage_lock_timeline.py`:

  signature  : sha256 over (owner script content sig + every input file's
               hash + params JSON). Identical signature => output is still
               valid => stage is LOCKED (skip on next run).
  outputs    : {path: {exists, bytes, mtime}}
  ran_at     : epoch seconds of the last successful run
  rules      : operator-rule strings — mirrored into the manifest so the run
               itself documents what the operator agreed to
  history    : past runs. Outputs of past runs are archived under
               `<run_dir>/attic/<stage>/<signature>/` (full byte copies).

A stage is LOCKED iff:
  - the manifest has an entry with signature == current computed signature
  - every declared output exists and is non-empty

If any of those fail, the stage is STALE (or MISSING) and `run_stage`
executes it. Before the rerun, current outputs are archived to the attic and
the previous manifest entry is pushed onto the history array — so the
timeline is always restorable (`restore_outputs` / `rollback_stage`).

Why input hashing (not just file existence): caching keyed on a marker file
lies whenever the owner script changes or an upstream output is replaced with
new content at the same path. Content hashing makes "same bytes -> same
signature" true regardless of filesystem metadata (`touch`, scp -p).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CHUNK_HASH_THRESHOLD_BYTES = 256 * 1024 * 1024  # hash huge files in strides
ATTIC_PRUNE_KEEP = 5  # history entries kept per stage (FIFO); 0 = forever


# ---------------------------------------------------------------------------
# specs
# ---------------------------------------------------------------------------
@dataclass
class InputSpec:
    """One declared input. `produced_by` names the upstream stage that owns
    the file (None for scripts / operator-provided data)."""

    path: Path
    produced_by: str | None = None


@dataclass
class StageSpec:
    """One stage: who produces it, what it reads/writes, in what order."""

    stage_id: str
    owner: Path
    outputs: list[Path]
    depends_on: list[str] = field(default_factory=list)
    inputs: list[InputSpec] = field(default_factory=list)
    phase: str | None = None  # optional CLI arg forwarded to the owner
    params: dict[str, Any] = field(default_factory=dict)
    rules: list[str] = field(default_factory=list)


@dataclass
class StageStatus:
    stage_id: str
    state: str  # 'locked' | 'stale' | 'restored' | 'missing' | 'unknown'
    reason: str
    outputs_ok: bool
    last_ran_at: float | None


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------
def _file_hash(path: Path) -> str:
    """Content hash of a file. Small files: full sha256. Huge files: strided
    sha256 + size (keeps a 43-min source probe under ~1 s)."""
    if not path.exists():
        return "MISSING"
    size = path.stat().st_size
    if size == 0:
        return "EMPTY"
    h = hashlib.sha256()
    if size <= CHUNK_HASH_THRESHOLD_BYTES:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
                h.update(chunk)
        return f"sha256:{h.hexdigest()[:24]}"
    n_strides = 64
    stride = max(1, size // n_strides)
    h.update(f"size={size}".encode())
    with path.open("rb") as f:
        for i in range(n_strides):
            f.seek(i * stride)
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
    return f"strided:{h.hexdigest()[:24]}"


def _owner_signature(path: Path) -> str:
    """Content signature for a script file: git commit if tracked, else full
    sha256. NEVER mtime+size — any `touch` would flip it (the old bug)."""
    if not path.exists():
        return "MISSING"
    try:
        r = subprocess.run(
            ["git", "-C", str(path.parent), "log", "-1", "--format=%H", "--", str(path)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return f"git:{r.stdout.strip()[:12]}"
    except Exception:
        pass
    return f"sha256:{_file_hash(path)}"


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------
class StageLock:
    """Manifest-backed stage lock over a caller-supplied stage graph.

    `root`    — filesystem root all spec paths are resolved against
    `run_dir` — where `stage_manifest.json` and `attic/` live
    """

    def __init__(self, graph: dict[str, StageSpec], root: Path, run_dir: Path):
        self.graph = graph
        self.root = Path(root)
        self.manifest_path = Path(run_dir) / "stage_manifest.json"
        self.attic_root = Path(run_dir) / "attic"

    # -- graph helpers -----------------------------------------------------
    def downstream_of(self, stage_id: str) -> list[str]:
        """Transitive closure of stages that depend on `stage_id`."""
        out: set[str] = set()
        changed = True
        while changed:
            changed = False
            for sid, spec in self.graph.items():
                if sid in out or sid == stage_id:
                    continue
                if any(dep == stage_id for dep in spec.depends_on) or any(
                    dep in out for dep in spec.depends_on
                ):
                    out.add(sid)
                    changed = True
        return sorted(out)

    def topological_order(self) -> list[str]:
        order: list[str] = []
        seen: set[str] = set()

        def visit(sid: str) -> None:
            if sid in seen:
                return
            for dep in self.graph[sid].depends_on:
                visit(dep)
            seen.add(sid)
            order.append(sid)

        for sid in self.graph:
            visit(sid)
        return order

    def _resolve(self, p: Path) -> Path:
        return p if p.is_absolute() else self.root / p

    # -- signature ---------------------------------------------------------
    def compute_signature(self, stage_id: str) -> str:
        spec = self.graph[stage_id]
        parts: list[str] = []
        owner_path = self._resolve(spec.owner)
        parts.append(f"owner={spec.owner}@{_owner_signature(owner_path)}")
        for inp in spec.inputs:
            p = self._resolve(inp.path)
            if inp.produced_by is None and p.suffix in (".py", ".sh"):
                parts.append(f"script:{inp.path}@{_owner_signature(p)}")
            else:
                parts.append(f"data:{inp.path}={_file_hash(p)}")
        parts.append(f"params={json.dumps(spec.params, sort_keys=True)}")
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    # -- manifest I/O --------------------------------------------------------
    def load_manifest(self) -> dict[str, dict[str, Any]]:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text())
            except Exception:
                return {}
        return {}

    def save_manifest(self, data: dict[str, dict[str, Any]]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True))

    # -- status --------------------------------------------------------------
    def status_of(self, stage_id: str, manifest: dict | None = None) -> StageStatus:
        manifest = manifest or self.load_manifest()
        spec = self.graph[stage_id]
        sig = self.compute_signature(stage_id)
        entry = manifest.get(stage_id)
        outputs_ok = all(
            self._resolve(o).exists() and self._resolve(o).stat().st_size > 0 for o in spec.outputs
        )
        if entry is None:
            return StageStatus(stage_id, "unknown", "never run", outputs_ok, None)
        if entry.get("signature") != sig:
            # "Deliberately restored to a past run" vs "accidentally stale
            # because an input changed" — same on disk, different intent.
            if entry.get("restored_to"):
                return StageStatus(
                    stage_id,
                    "restored",
                    f"outputs restored to past run sig={entry['signature'][:8]} "
                    f"(current inputs would produce sig={sig[:8]})",
                    outputs_ok,
                    entry.get("ran_at"),
                )
            return StageStatus(
                stage_id,
                "stale",
                "inputs/owner changed since last run",
                outputs_ok,
                entry.get("ran_at"),
            )
        if not outputs_ok:
            return StageStatus(
                stage_id,
                "missing",
                "output files missing or empty",
                outputs_ok,
                entry.get("ran_at"),
            )
        return StageStatus(
            stage_id,
            "locked",
            "signature matches + outputs present",
            outputs_ok,
            entry.get("ran_at"),
        )

    def all_status(self) -> list[StageStatus]:
        m = self.load_manifest()
        return [self.status_of(sid, m) for sid in self.topological_order()]

    # -- attic (timeline archive) ---------------------------------------------
    def _attic_dir(self, stage_id: str, signature: str) -> Path:
        return self.attic_root / stage_id / signature

    def _archive_copy(self, src: Path, dst: Path) -> None:
        """Full byte copy — NEVER a hardlink. A hardlink shares the inode, so
        an owner that overwrites its output in place would silently corrupt
        the attic archive too (regression pinned by test_archive_survives...)."""
        import shutil

        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            s_src, s_dst = src.stat(), dst.stat()
            if s_src.st_size == s_dst.st_size and abs(s_src.st_mtime - s_dst.st_mtime) < 1:
                return
        shutil.copy2(src, dst)

    def archive_current_outputs(self, stage_id: str, manifest_entry: dict[str, Any]) -> Path | None:
        """Snapshot CURRENT outputs under their recorded signature. Called
        BEFORE a rerun overwrites them. Idempotent."""
        spec = self.graph[stage_id]
        sig = manifest_entry.get("signature")
        if not sig:
            return None
        attic = self._attic_dir(stage_id, sig)
        already = (
            all((attic / Path(o).name).exists() for o in spec.outputs) if attic.exists() else False
        )
        if already:
            return attic
        attic.mkdir(parents=True, exist_ok=True)
        archived = []
        for o in spec.outputs:
            p = self._resolve(o)
            if p.exists():
                self._archive_copy(p, attic / p.name)
                archived.append(p.name)
        if not archived:
            with contextlib.suppress(OSError):
                attic.rmdir()
            return None
        return attic

    def _prune_history(self, stage_id: str, manifest_entry: dict[str, Any]) -> None:
        if ATTIC_PRUNE_KEEP <= 0:
            return
        history = manifest_entry.get("history") or []
        if len(history) <= ATTIC_PRUNE_KEEP:
            return
        to_drop = history[: len(history) - ATTIC_PRUNE_KEEP]
        for entry in to_drop:
            attic = entry.get("archived_to")
            if not attic:
                continue
            attic_path = Path(attic) if Path(attic).is_absolute() else self.root / attic
            if attic_path.exists():
                import shutil

                shutil.rmtree(attic_path, ignore_errors=True)
        manifest_entry["history"] = history[len(history) - ATTIC_PRUNE_KEEP :]

    def _record_outputs(self, spec: StageSpec) -> dict[str, dict[str, Any]]:
        record = {}
        for o in spec.outputs:
            p = self._resolve(o)
            record[str(o)] = {
                "exists": p.exists(),
                "bytes": p.stat().st_size if p.exists() else 0,
                "mtime": p.stat().st_mtime if p.exists() else 0,
            }
        return record

    # -- timeline commands (the video-editor metaphor) -------------------------
    def restore_outputs(self, stage_id: str, target_sig: str) -> StageStatus:
        """Restore a past run's outputs from the attic. CURRENT outputs are
        archived first, so you can always restore-forward again."""
        if stage_id not in self.graph:
            raise KeyError(f"unknown stage: {stage_id}")
        spec = self.graph[stage_id]
        manifest = self.load_manifest()
        entry = manifest.get(stage_id)
        if entry is None:
            return StageStatus(
                stage_id, "unknown", "never run — nothing to restore from", False, None
            )

        cur_sig = entry.get("signature")
        if target_sig == cur_sig:
            return self.status_of(stage_id, manifest)

        target_attic = self._attic_dir(stage_id, target_sig)
        if not target_attic.exists():
            return StageStatus(
                stage_id,
                "missing",
                f"attic {target_attic} not found — no archived copy to restore",
                False,
                entry.get("ran_at"),
            )

        cur_attic = self.archive_current_outputs(stage_id, entry)
        if cur_attic and cur_sig:
            history = entry.setdefault("history", [])
            if not any(h.get("signature") == cur_sig for h in history):
                history.append(
                    {
                        "signature": cur_sig,
                        "ran_at": entry.get("ran_at"),
                        "archived_to": str(cur_attic),
                        "archived_at": time.time(),
                        "restored_away_for": target_sig,
                    }
                )
            self._prune_history(stage_id, entry)

        for o in spec.outputs:
            p = self._resolve(o)
            attic_p = target_attic / p.name
            if not attic_p.exists():
                return StageStatus(
                    stage_id,
                    "missing",
                    f"attic missing output {attic_p}",
                    False,
                    entry.get("ran_at"),
                )
            with contextlib.suppress(FileNotFoundError):
                p.unlink()
            self._archive_copy(attic_p, p)

        entry["signature"] = target_sig
        entry["ran_at"] = time.time()
        entry["restored_to"] = target_sig
        entry["restored_at"] = time.time()
        entry["outputs"] = self._record_outputs(spec)
        manifest[stage_id] = entry
        self.save_manifest(manifest)
        return self.status_of(stage_id, manifest)

    def rollback_stage(self, stage_id: str) -> StageStatus:
        """Undo the last rerun (restore the most recent history entry)."""
        manifest = self.load_manifest()
        entry = manifest.get(stage_id)
        if entry is None:
            return StageStatus(stage_id, "unknown", "never run — nothing to roll back", False, None)
        history = entry.get("history") or []
        if not history:
            return StageStatus(
                stage_id,
                "missing",
                "history empty — nothing to roll back",
                False,
                entry.get("ran_at"),
            )
        target = history[-1]
        return self.restore_outputs(stage_id, target["signature"])

    def stage_history(self, stage_id: str) -> list[dict[str, Any]]:
        manifest = self.load_manifest()
        entry = manifest.get(stage_id)
        if entry is None:
            return []
        cur_sig = entry.get("signature")
        return [
            {
                "n": i + 1,
                "signature": h.get("signature"),
                "ran_at": h.get("ran_at"),
                "archived_to": h.get("archived_to"),
                "archived_at": h.get("archived_at"),
                "is_current": h.get("signature") == cur_sig,
            }
            for i, h in enumerate(entry.get("history") or [])
        ]

    # -- execution -------------------------------------------------------------
    def _invoke_owner(self, stage_id: str) -> int:
        """Run the owner script. Returns process exit code."""
        spec = self.graph[stage_id]
        owner = self._resolve(spec.owner)
        if not owner.exists():
            print(f"  [lock] owner script missing: {owner}", file=sys.stderr)
            return 127
        if owner.suffix == ".sh":
            cmd = ["bash", str(owner)]
        else:
            cmd = [sys.executable, str(owner)]
        if spec.phase:
            cmd.append(spec.phase)
        env = os.environ.copy()
        env_file = self.root / ".env"
        if env_file.exists():
            for ln in env_file.read_text().splitlines():
                if ln and not ln.startswith("#") and "=" in ln:
                    k, v = ln.split("=", 1)
                    env.setdefault(k.strip(), v.strip())
        r = subprocess.run(cmd, cwd=str(self.root), env=env)
        return r.returncode

    def run_stage(self, stage_id: str, force: bool = False) -> StageStatus:
        """Run `stage_id` iff not locked (or force=True). Records the manifest
        entry; archives the previous run into the attic first."""
        if stage_id not in self.graph:
            raise KeyError(f"unknown stage: {stage_id}")
        manifest = self.load_manifest()
        cur = self.status_of(stage_id, manifest)
        if cur.state == "locked" and not force:
            return cur
        prev_entry = manifest.get(stage_id)
        prev_attic = None
        if prev_entry is not None and prev_entry.get("signature"):
            prev_attic = self.archive_current_outputs(stage_id, prev_entry)
        rc = self._invoke_owner(stage_id)
        if rc != 0:
            # manifest NOT updated: a failed run must never look locked
            return self.status_of(stage_id, manifest)
        spec = self.graph[stage_id]
        sig = self.compute_signature(stage_id)
        history = []
        if prev_entry is not None:
            history = list(prev_entry.get("history") or [])
            prev_sig = prev_entry.get("signature")
            if (
                prev_sig
                and prev_attic is not None
                and not any(h.get("signature") == prev_sig for h in history)
            ):
                history.append(
                    {
                        "signature": prev_sig,
                        "ran_at": prev_entry.get("ran_at"),
                        "archived_to": str(prev_attic),
                        "archived_at": time.time(),
                        "superseded_by": sig,
                    }
                )
        manifest[stage_id] = {
            "signature": sig,
            "outputs": self._record_outputs(spec),
            "ran_at": time.time(),
            "rules": spec.rules,
            "history": history,
        }
        self._prune_history(stage_id, manifest[stage_id])
        self.save_manifest(manifest)
        return self.status_of(stage_id, manifest)

    def run_from(self, stage_id: str, force: bool = False) -> list[StageStatus]:
        """Run `stage_id` + every downstream stage. Upstream stages are NOT
        touched — fix the broken stage and its consumers, never the world."""
        targets = [stage_id] + self.downstream_of(stage_id)
        results = []
        for tid in targets:
            is_target = tid == stage_id
            results.append(self.run_stage(tid, force=force if is_target else True))
            if results[-1].state not in ("locked", "restored"):
                break
        return results

    def lock_stage(self, stage_id: str) -> StageStatus:
        """Record the current signature WITHOUT running the owner — bootstrap
        the manifest from already-approved outputs. Refuses when outputs are
        missing (never trust a lock without its artifacts)."""
        if stage_id not in self.graph:
            raise KeyError(f"unknown stage: {stage_id}")
        spec = self.graph[stage_id]
        missing = [
            str(o)
            for o in spec.outputs
            if not self._resolve(o).exists() or self._resolve(o).stat().st_size == 0
        ]
        if missing:
            return StageStatus(stage_id, "missing", f"outputs missing: {missing}", False, None)
        sig = self.compute_signature(stage_id)
        manifest = self.load_manifest()
        prev = manifest.get(stage_id, {})
        manifest[stage_id] = {
            "signature": sig,
            "outputs": self._record_outputs(spec),
            "ran_at": time.time(),
            "rules": spec.rules,
            "locked_without_rerun": True,
            "history": list(prev.get("history") or []),
        }
        self.save_manifest(manifest)
        return self.status_of(stage_id, manifest)

    def invalidate(self, stage_id: str) -> bool:
        """Mark a stage stale: drop its manifest entry. Returns True if a
        manifest entry was removed."""
        manifest = self.load_manifest()
        if stage_id in manifest:
            manifest.pop(stage_id)
            self.save_manifest(manifest)
            return True
        return False

    def invalidate_downstream(self, stage_id: str) -> list[str]:
        """Drop manifest entries for `stage_id` and everything downstream."""
        manifest = self.load_manifest()
        dropped = []
        for tid in [stage_id] + self.downstream_of(stage_id):
            if tid in manifest:
                manifest.pop(tid)
                dropped.append(tid)
        self.save_manifest(manifest)
        return dropped
