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
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from director_os.core.http import build_headers
from director_os.perception.transcript import Transcript, TranscriptSegment, WordTiming

DEFAULT_POLL_INTERVAL_S = 2.0
DEFAULT_TIMEOUT_S = 600.0
# The 2026-11-23 incident law: a 43-min bodycam's audio is ~20 MB and uploads
# over home upstream for minutes. The old flat ``timeout=60`` killed the
# request mid-body ("Server disconnected without sending a response") and
# Gladia's own edge answered 408 on slow bodies. Uploads get a generous
# write window; connect stays tight so a dead DNS fails fast.
DEFAULT_HTTP_TIMEOUT_S = 900.0
DEFAULT_CONNECT_TIMEOUT_S = 30.0
UPLOAD_ATTEMPTS = 3
# Transient provider/network failures worth a retry with backoff. Auth (401/403)
# and most 4xx are permanent — they must fail loudly, never burn retries.
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


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
        log: Callable[[str], None] | None = None,
    ):
        if not api_key:
            raise GladiaError("GLADIA_API_KEY is not configured")
        self._api_key = api_key
        self._base = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._poll_interval_s = poll_interval_s
        self._contact_email = contact_email
        self._log = log or (lambda _msg: None)
        # Long-body upload law: 900s total with a tight connect — the flat
        # 60s default used to abort large audio uploads mid-flight.
        self._http = http or httpx.Client(
            timeout=httpx.Timeout(DEFAULT_HTTP_TIMEOUT_S, connect=DEFAULT_CONNECT_TIMEOUT_S)
        )

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

        audio_url = self.upload(audio_path)
        job_id = self.create_job(audio_url, language=language)
        payload = self.fetch_result(job_id)
        return self.normalize(payload, asset_id)

    # --- public primitives (the ASR cache composes these so a retry adopts an
    # existing transcription instead of re-uploading the same bytes) ---
    def upload(self, audio_path: Path) -> str:
        """`POST /v2/upload` -> {audio_url}."""
        return self._upload(Path(audio_path))

    def create_job(self, audio_url: str, language: str | None = None) -> str:
        """`POST /v2/transcription` -> {id} (async job)."""
        return self._create_job(audio_url, language=language)

    def fetch_result(self, transcription_id: str) -> dict:
        """Poll an EXISTING transcription id until done; return the raw payload.

        This is the 'gladia already has it' path: given a persisted id, the
        finished result is fetched without any new upload.
        """
        return self._poll_result(str(transcription_id))

    def normalize(self, payload: dict, asset_id: str) -> Transcript:
        """Provider payload -> canonical `Transcript` (reusable per asset)."""
        return self._normalize(payload, asset_id)

    def _upload(self, audio_path: Path) -> str:
        mib = audio_path.stat().st_size / (1024 * 1024)
        self._log(
            f"  [gladia] uploading {audio_path.name} ({mib:.1f} MiB) — "
            f"up to {UPLOAD_ATTEMPTS} attempt(s)"
        )
        last_err: Exception | None = None
        for attempt in range(1, UPLOAD_ATTEMPTS + 1):
            try:
                with open(audio_path, "rb") as f:
                    files = {"audio": (audio_path.name, f, "application/octet-stream")}
                    resp = self._http.post(
                        f"{self._base}/v2/upload",
                        headers=build_headers(self._contact_email)
                        | {"x-gladia-key": self._api_key},
                        files=files,
                    )
                if resp.status_code in RETRYABLE_STATUS:
                    # provider edge timed out / throttled the body — back off
                    # and resend; the failure text mirrors what the Studio saw
                    raise GladiaError(f"gladia upload transient HTTP {resp.status_code} — retrying")
                self._raise_for_status(resp, "upload")
                audio_url = resp.json().get("audio_url")
                if not audio_url:
                    raise GladiaError(f"upload response missing audio_url: {resp.text[:300]}")
                return audio_url
            except (httpx.TransportError, GladiaError) as e:
                permanent = (
                    isinstance(e, GladiaError)
                    and "transient" not in str(e)
                    and "missing audio_url" not in str(e)
                )
                if permanent or attempt == UPLOAD_ATTEMPTS:
                    raise GladiaError(
                        f"gladia upload failed after {attempt} attempt(s): {e}"
                    ) from e
                last_err = e
                delay = 4.0 * 2 ** (attempt - 1)
                self._log(
                    f"  [gladia] upload attempt {attempt}/{UPLOAD_ATTEMPTS} failed ({e}) "
                    f"— retrying in {delay:.0f}s"
                )
                time.sleep(delay)
        raise GladiaError(f"gladia upload failed: {last_err}")

    def _create_job(self, audio_url: str, language: str | None = None) -> str:
        # v2 API (2026-11): `language_behaviour` was removed — `language` is
        # optional and automatic multi-language detection is the default.
        body: dict[str, Any] = {"audio_url": audio_url, "diarization": True}
        if language:
            body["language"] = language
        resp = self._request_with_retry(
            "POST", f"{self._base}/v2/transcription", "create job", json_body=body
        )
        job_id = resp.get("id")
        if not job_id:
            raise GladiaError(f"transcription response missing job id: {str(resp)[:300]}")
        return str(job_id)

    def _poll_result(self, job_id: str) -> dict:
        deadline = time.monotonic() + self._timeout_s
        url = f"{self._base}/v2/transcription/{job_id}"
        last_status: str | None = None
        while True:
            data = self._request_with_retry("GET", url, f"poll {job_id}")
            status = data.get("status")
            if status != last_status:
                # one console line per state change — a 43-min transcription
                # sits in 'processing' for many minutes and silence reads as death
                self._log(f"  [gladia] job {job_id[:8]}… status: {status or 'unknown'}")
                last_status = status
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

    def _request_with_retry(
        self,
        method: str,
        url: str,
        what: str,
        json_body: dict[str, Any] | None = None,
        attempts: int = 3,
    ) -> dict:
        """One provider request with small retries — a dropped connection must
        never kill a transcription that already paid its upload."""
        last_err: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                resp = self._http.request(method, url, headers=self._headers(), json=json_body)
                if resp.status_code in RETRYABLE_STATUS:
                    raise GladiaError(f"gladia {what} transient HTTP {resp.status_code}")
                self._raise_for_status(resp, what)
                return resp.json()
            except (httpx.TransportError, GladiaError) as e:
                permanent = isinstance(e, GladiaError) and "transient" not in str(e)
                if permanent or attempt == attempts:
                    raise GladiaError(
                        f"gladia {what} failed after {attempt} attempt(s): {e}"
                    ) from e
                last_err = e
                delay = 2.0 * 2 ** (attempt - 1)
                self._log(f"  [gladia] {what} attempt {attempt}/{attempts} failed ({e}) — retrying")
                time.sleep(delay)
        raise GladiaError(f"gladia {what} failed: {last_err}")

    # ------------------------------------------------------------ normalize
    def _normalize(self, payload: dict, asset_id: str) -> Transcript:
        result = payload.get("result") or {}
        transcription = result.get("transcription") or {}

        raw_words = transcription.get("word_level_results") or transcription.get("words") or []
        if not raw_words:
            # v2 API (2026-11): word timings ride inside utterances — flatten them,
            # inheriting each utterance's diarized speaker onto its words.
            for utt in transcription.get("utterances") or []:
                speaker = utt.get("speaker")
                for w in utt.get("words") or []:
                    # speaker "0" is a valid label and falsy — test None, not truthiness
                    if speaker is not None and not w.get("speaker"):
                        w["speaker"] = speaker
                    raw_words.append(w)
        if not raw_words:
            raise GladiaError(
                f"gladia response has no word-level results (keys: {sorted(transcription.keys())})"
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
            raise GladiaError(f"gladia {what} failed: HTTP {resp.status_code}: {resp.text[:300]}")


def _to_seconds(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _speaker(w: dict) -> str | None:
    # speaker "0" is a real label — never fall through on falsy zero
    speaker = w.get("speaker")
    if speaker is None:
        speaker = w.get("speaker_label") or w.get("diarization_speaker")
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
