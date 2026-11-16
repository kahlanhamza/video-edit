"""Gladia ASR adapter — cloud speech-to-text with diarization (PLAN §4.5).

Implements the Gladia v2 flow and normalizes into the perception `Transcript`
contract, so downstream stages never see provider quirks:

1. `POST /v2/upload`            -> {audio_url}
2. `POST /v2/transcription`     -> {id}  (async job)
3. `GET  /v2/transcription/{id}`-> poll until done -> word-level results

Guardrail 3 applies: every outbound request carries the descriptive
DirectorOS User-Agent (built by `core.http.build_headers`).

Field normalization is defensive (gladia evolves its schema): words are read
from `word_level_results` or `words`, seconds from `start`/`end`, speaker from
`speaker`. Unknown shapes raise `GladiaError` instead of silently producing an
empty transcript.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx
from director_os.core.http import build_headers
from director_os.perception.transcript import Transcript, TranscriptSegment, WordTiming

DEFAULT_POLL_INTERVAL_S = 2.0
DEFAULT_TIMEOUT_S = 600.0


class GladiaError(RuntimeError):
    pass


class GladiaClient:
    """Async-job client; inject `http` for tests (httpx.MockTransport)."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.gladia.io",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        http: httpx.Client | None = None,
        contact_email: str | None = None,
    ):
        if not api_key:
            raise GladiaError("GLADIA_API_KEY is not configured")
        self._api_key = api_key
        self._base = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._poll_interval_s = poll_interval_s
        self._contact_email = contact_email
        self._http = http or httpx.Client(timeout=60.0)

    # ------------------------------------------------------------ headers
    def _headers(self, json_body: bool = True) -> dict[str, str]:
        headers = build_headers(self._contact_email)
        headers["x-gladia-key"] = self._api_key
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    # ------------------------------------------------------------ flow
    def transcribe(
        self, audio_path: Path, asset_id: str, language: str | None = None
    ) -> Transcript:
        """Full flow: upload -> create job -> poll -> normalize."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise GladiaError(f"audio file missing: {audio_path}")

        audio_url = self._upload(audio_path)
        job_id = self._create_job(audio_url, language=language)
        payload = self._poll_result(job_id)
        return self._normalize(payload, asset_id)

    def _upload(self, audio_path: Path) -> str:
        with open(audio_path, "rb") as f:
            files = {"audio": (audio_path.name, f, "application/octet-stream")}
            resp = self._http.post(
                f"{self._base}/v2/upload",
                headers=build_headers(self._contact_email) | {"x-gladia-key": self._api_key},
                files=files,
            )
        self._raise_for_status(resp, "upload")
        audio_url = resp.json().get("audio_url")
        if not audio_url:
            raise GladiaError(f"upload response missing audio_url: {resp.text[:300]}")
        return audio_url

    def _create_job(self, audio_url: str, language: str | None = None) -> str:
        body: dict[str, Any] = {
            "audio_url": audio_url,
            "diarization": True,
            "language_behaviour": "single language" if language else "automatic multi language",
        }
        if language:
            body["language"] = language
        resp = self._http.post(
            f"{self._base}/v2/transcription", headers=self._headers(), json=body
        )
        self._raise_for_status(resp, "create job")
        data = resp.json()
        job_id = data.get("id")
        if not job_id:
            raise GladiaError(f"transcription response missing job id: {resp.text[:300]}")
        return str(job_id)

    def _poll_result(self, job_id: str) -> dict:
        deadline = time.monotonic() + self._timeout_s
        url = f"{self._base}/v2/transcription/{job_id}"
        while True:
            resp = self._http.get(url, headers=self._headers())
            self._raise_for_status(resp, f"poll {job_id}")
            data = resp.json()
            status = data.get("status")
            if status == "done":
                return data
            if status in ("error", "failed"):
                raise GladiaError(f"gladia job {job_id} failed: {str(data)[:400]}")
            if time.monotonic() > deadline:
                raise GladiaError(
                    f"gladia job {job_id} timed out after {self._timeout_s:.0f}s "
                    f"(last status: {status})"
                )
            time.sleep(self._poll_interval_s)

    # ------------------------------------------------------------ normalize
    def _normalize(self, payload: dict, asset_id: str) -> Transcript:
        result = payload.get("result") or {}
        transcription = result.get("transcription") or {}

        raw_words = transcription.get("word_level_results") or transcription.get("words") or []
        if not raw_words:
            raise GladiaError(
                "gladia response has no word-level results "
                f"(keys: {sorted(transcription.keys())})"
            )

        words: list[WordTiming] = []
        for w in raw_words:
            text = (w.get("word") or w.get("text") or "").strip()
            start = _to_seconds(w.get("start"))
            end = _to_seconds(w.get("end"))
            if not text or start is None or end is None:
                continue  # skip malformed entries rather than fail the whole job
            words.append(
                WordTiming(
                    text=text,
                    start_s=start,
                    end_s=max(end, start + 0.01),
                    speaker=_speaker(w),
                    confidence=float(w.get("confidence") or 1.0),
                )
            )
        if not words:
            raise GladiaError("gladia word-level results all malformed")

        segments = _group_utterances(transcription, words)
        languages = transcription.get("languages") or []
        duration = max(w.end_s for w in words)

        return Transcript(
            asset_id=asset_id,
            language=(languages[0] if languages else "en"),
            duration_s=duration,
            words=words,
            segments=segments,
            provider="gladia",
        )

    def _raise_for_status(self, resp: httpx.Response, what: str) -> None:
        if resp.status_code in (401, 403):
            raise GladiaError(f"gladia auth rejected during {what} ({resp.status_code})")
        if resp.status_code == 429:
            raise GladiaError(f"gladia rate-limited during {what} (429)")
        if resp.status_code >= 400:
            raise GladiaError(
                f"gladia {what} failed: HTTP {resp.status_code}: {resp.text[:300]}"
            )


def _to_seconds(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _speaker(w: dict) -> str | None:
    speaker = w.get("speaker") or w.get("speaker_label") or w.get("diarization_speaker")
    return str(speaker) if speaker is not None else None


def _group_utterances(transcription: dict, words: list[WordTiming]) -> list[TranscriptSegment]:
    """Group word timings into diarized segments (by speaker runs)."""
    segments: list[TranscriptSegment] = []
    current_speaker: str | None = None
    run: list[WordTiming] = []

    for w in words:
        if w.speaker != current_speaker and run:
            segments.append(_build_segment(current_speaker or "UNKNOWN", run))
            run = []
        current_speaker = w.speaker
        run.append(w)
    if run:
        segments.append(_build_segment(current_speaker or "UNKNOWN", run))
    return segments


def _build_segment(speaker: str, run: list[WordTiming]) -> TranscriptSegment:
    return TranscriptSegment(
        speaker=speaker,
        start_s=run[0].start_s,
        end_s=run[-1].end_s,
        text=" ".join(w.text for w in run),
        words=list(run),
    )
