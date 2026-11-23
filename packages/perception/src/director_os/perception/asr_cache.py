"""Content-addressed ASR cache — never upload the same bytes twice.

The owner law (2026-11-23): if the SAME media was already transcribed, the
provider already has the result — we only need a way to GET it. Retries and
re-ingests of identical content must not re-upload (re-billed, re-waited).

Flow (drop-in replacement for the bare ``GladiaClient.transcribe`` callable
the pipeline expects as ``deps.transcribe(path, asset_id)``):

1. sha256 the media file (the exact bytes that WOULD be uploaded)
2. local hit (status=done + normalized payload) -> pure cache return, no network
3. remote hit (status=processing + external id) -> poll THAT id, no upload —
   a crash or poll-timeout mid-transcription adopts the same id on the next run
4. miss (or a provider-side error recorded earlier) -> upload + create job,
   persist the id as `processing` BEFORE polling (crash-safe), then poll

Provider-side failure (status=error/failed on the remote job) marks the row
`error` so the next attempt starts a fresh upload; a poll TIMEOUT leaves the
row `processing` so the next attempt adopts the id and keeps waiting.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from director_os.core.db import ASRCacheRow
from director_os.perception.gladia import GladiaClient, GladiaError
from director_os.perception.transcript import Transcript

GLADIA_PROVIDER = "gladia"
_CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Streamed content digest — the upload-avoidance key."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(_CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


class CachedTranscriber:
    """Wraps a ``GladiaClient`` with the (provider, sha256) dedup layer."""

    def __init__(self, client: GladiaClient, sessions: Any, log: Any = None):
        self._client = client
        self._sessions = sessions  # sessionmaker (pipeline deps.sessions)
        self._log = log or (lambda _msg: None)

    # ------------------------------------------------------------------ reads
    def _get(self, digest: str) -> ASRCacheRow | None:
        with self._sessions() as s:
            return s.get(ASRCacheRow, (GLADIA_PROVIDER, digest))

    def _put(
        self,
        digest: str,
        *,
        status: str,
        external_id: str | None = None,
        transcript: dict | None = None,
        language: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._sessions() as s:
            row = s.get(ASRCacheRow, (GLADIA_PROVIDER, digest))
            if row is None:
                row = ASRCacheRow(provider=GLADIA_PROVIDER, content_sha256=digest, status=status)
                s.add(row)
            row.status = status
            row.external_id = external_id if external_id is not None else row.external_id
            row.transcript_json = transcript if transcript is not None else row.transcript_json
            row.language = language if language is not None else row.language
            row.error = error
            row.updated_at = datetime.now(UTC)
            s.commit()

    # ------------------------------------------------------------------ flow
    def transcribe(self, media_path: Path, asset_id: str) -> Transcript:
        media_path = Path(media_path)
        if not media_path.exists():
            raise GladiaError(f"media file missing: {media_path}")

        digest = sha256_file(media_path)
        row = self._get(digest)

        # 2) pure local hit — zero network
        if row is not None and row.status == "done" and row.transcript_json:
            self._log(f"  [asr-cache] hit {digest[:12]} — reusing stored transcription")
            return self._rebind(row.transcript_json, asset_id, row.language)

        # 3) remote hit — Gladia already has (or is still building) this job
        if row is not None and row.status == "processing" and row.external_id:
            self._log(
                f"  [asr-cache] adopting gladia job {row.external_id} for {digest[:12]} "
                "— no re-upload"
            )
            try:
                payload = self._client.fetch_result(row.external_id)
            except GladiaError as e:
                if "failed" in str(e) or "error" in str(e).lower():
                    # provider-side failure: burn the row so the next attempt
                    # starts a fresh upload instead of polling a dead id
                    self._put(
                        digest, status="error", external_id=row.external_id, error=str(e)[:1000]
                    )
                raise
            return self._finish(digest, payload, asset_id)

        # 4) miss — upload, then record the id BEFORE polling (crash-safe)
        self._log(f"  [asr-cache] miss {digest[:12]} — uploading to gladia once")
        audio_url = self._client.upload(media_path)
        external_id = self._client.create_job(audio_url)
        self._put(digest, status="processing", external_id=external_id)
        try:
            payload = self._client.fetch_result(external_id)
        except GladiaError as e:
            if "failed" in str(e) or "error" in str(e).lower():
                self._put(digest, status="error", external_id=external_id, error=str(e)[:1000])
            # poll TIMEOUT leaves the row `processing`: the next run adopts
            # the same id and keeps waiting — the upload is never repeated
            raise
        return self._finish(digest, payload, asset_id)

    def _finish(self, digest: str, payload: dict, asset_id: str) -> Transcript:
        transcript = self._client.normalize(payload, asset_id)
        self._put(
            digest,
            status="done",
            transcript=transcript.model_dump(mode="json"),
            language=transcript.language,
        )
        return transcript

    @staticmethod
    def _rebind(cached: dict, asset_id: str, language: str | None) -> Transcript:
        """Reuse a stored normalized payload for a (possibly different) asset."""
        return Transcript.model_validate({**cached, "asset_id": asset_id})
