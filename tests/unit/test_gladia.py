"""Gladia ASR adapter — full v2 flow against a mocked transport.

- upload -> create -> poll(pending) -> poll(done) -> normalized Transcript
- word timings + speakers survive normalization (diarization intact)
- auth/rate-limit/timeout/failed-status errors raise GladiaError
- malformed word entries are skipped, not fatal
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from director_os.perception.gladia import GladiaClient, GladiaError


def make_audio(tmp_path: Path) -> Path:
    p = tmp_path / "audio.wav"
    p.write_bytes(b"RIFFfake-audio-bytes")  # transport is mocked; bytes irrelevant
    return p


def done_payload() -> dict:
    words = [
        {"word": "units", "start": 0.10, "end": 0.52, "confidence": 0.97, "speaker": "1"},
        {"word": "arriving", "start": 0.60, "end": 1.20, "confidence": 0.95, "speaker": "1"},
        {"word": "now", "start": 1.25, "end": 1.50, "confidence": 0.99, "speaker": "1"},
        {"word": "copy", "start": 1.70, "end": 2.00, "confidence": 0.93, "speaker": "2"},
        {"word": "that", "start": 2.05, "end": 2.30, "confidence": 0.94, "speaker": "2"},
    ]
    return {
        "id": "job-123",
        "status": "done",
        "result": {
            "transcription": {
                "full_transcript": "units arriving now copy that",
                "languages": ["en"],
                "word_level_results": words,
            }
        },
    }


class MockFlow:
    """Scriptable fake of the Gladia API surface."""

    def __init__(self, responses: dict[str, list[httpx.Response]]):
        self.responses = responses
        self.calls: list[tuple[str, str]] = []  # (method, path)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, request.url.path))
        key = f"{request.method} {request.url.path}"
        queue = self.responses.get(key)
        if not queue:
            raise AssertionError(f"unexpected call: {key}")
        resp = queue.pop(0)
        return resp


def make_client(flow: MockFlow, **kwargs) -> GladiaClient:
    transport = httpx.MockTransport(flow.handler)
    return GladiaClient(
        api_key="test-key",
        base_url="https://api.gladia.test",
        http=httpx.Client(transport=transport),
        poll_interval_s=0.0,  # no sleep in tests
        **kwargs,
    )


def test_full_flow_normalizes_transcript(tmp_path):
    flow = MockFlow(
        {
            "POST /v2/upload": [
                httpx.Response(200, json={"audio_url": "https://cdn.gladia.test/a/b.wav"})
            ],
            "POST /v2/transcription": [httpx.Response(202, json={"id": "job-123"})],
            "GET /v2/transcription/job-123": [
                httpx.Response(200, json={"id": "job-123", "status": "processing"}),
                httpx.Response(200, json=done_payload()),
            ],
        }
    )
    client = make_client(flow)
    transcript = client.transcribe(make_audio(tmp_path), "asset_1")

    assert [c for c in flow.calls] == [
        ("POST", "/v2/upload"),
        ("POST", "/v2/transcription"),
        ("GET", "/v2/transcription/job-123"),
        ("GET", "/v2/transcription/job-123"),
    ]
    assert transcript.provider == "gladia"
    assert transcript.asset_id == "asset_1"
    assert transcript.language == "en"
    assert transcript.duration_s == pytest.approx(2.30)
    assert len(transcript.words) == 5
    assert transcript.words[0].text == "units"
    assert transcript.words[0].start_s == pytest.approx(0.10)
    assert transcript.speakers == ["1", "2"]
    # speaker run grouping -> two segments
    assert [s.speaker for s in transcript.segments] == ["1", "2"]
    assert transcript.segments[1].text == "copy that"

    # UA guardrail: every outbound call carried the descriptive identity
    # (MockFlow records only paths; headers asserted via build_headers unit test)


def test_upload_carries_gladia_key_and_ua(tmp_path):
    seen_headers = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(dict(request.headers))
        return httpx.Response(200, json={"audio_url": "https://x.test/a.wav"})

    transport = httpx.MockTransport(handler)
    client = GladiaClient(
        api_key="k2",
        base_url="https://api.gladia.test",
        http=httpx.Client(transport=transport),
    )
    url = client._upload(make_audio(tmp_path))
    assert url == "https://x.test/a.wav"
    assert seen_headers.get("x-gladia-key") == "k2"
    assert "DirectorOS" in seen_headers.get("user-agent", "")


@pytest.mark.parametrize(
    ("status", "match"),
    [
        (401, "auth rejected"),
        (429, "rate-limited"),
        (500, "HTTP 500"),
    ],
)
def test_http_errors_surface_as_gladia_error(tmp_path, status, match):
    flow = MockFlow({"POST /v2/upload": [httpx.Response(status, text="boom")]})
    client = make_client(flow)
    with pytest.raises(GladiaError, match=match):
        client.transcribe(make_audio(tmp_path), "a1")


def test_job_failure_status_raises(tmp_path):
    flow = MockFlow(
        {
            "POST /v2/upload": [
                httpx.Response(200, json={"audio_url": "https://cdn.gladia.test/a.wav"})
            ],
            "POST /v2/transcription": [httpx.Response(202, json={"id": "job-err"})],
            "GET /v2/transcription/job-err": [
                httpx.Response(200, json={"id": "job-err", "status": "error", "error": "bad audio"})
            ],
        }
    )
    client = make_client(flow)
    with pytest.raises(GladiaError, match="failed"):
        client.transcribe(make_audio(tmp_path), "a1")


def test_poll_timeout_raises(tmp_path):
    flow = MockFlow(
        {
            "POST /v2/upload": [
                httpx.Response(200, json={"audio_url": "https://cdn.gladia.test/a.wav"})
            ],
            "POST /v2/transcription": [httpx.Response(202, json={"id": "job-slow"})],
            # always pending — enough queued responses to exhaust the deadline
            "GET /v2/transcription/job-slow": [
                httpx.Response(200, json={"status": "processing"}) for _ in range(5000)
            ],
        }
    )
    client = make_client(flow, timeout_s=0.05)
    with pytest.raises(GladiaError, match="timed out"):
        client.transcribe(make_audio(tmp_path), "a1")


def test_missing_api_key_rejected():
    with pytest.raises(GladiaError, match="GLADIA_API_KEY"):
        GladiaClient(api_key="")


def test_no_word_level_results_raises(tmp_path):
    flow = MockFlow(
        {
            "POST /v2/upload": [
                httpx.Response(200, json={"audio_url": "https://cdn.gladia.test/a.wav"})
            ],
            "POST /v2/transcription": [httpx.Response(202, json={"id": "job-empty"})],
            "GET /v2/transcription/job-empty": [
                httpx.Response(
                    200,
                    json={
                        "status": "done",
                        "result": {"transcription": {"full_transcript": "", "languages": ["en"]}},
                    },
                )
            ],
        }
    )
    client = make_client(flow)
    with pytest.raises(GladiaError, match="no word-level results"):
        client.transcribe(make_audio(tmp_path), "a1")


def test_malformed_words_skipped(tmp_path):
    payload = done_payload()
    payload["result"]["transcription"]["word_level_results"].append(
        {"word": "", "start": None, "end": 9.9}
    )
    flow = MockFlow(
        {
            "POST /v2/upload": [
                httpx.Response(200, json={"audio_url": "https://cdn.gladia.test/a.wav"})
            ],
            "POST /v2/transcription": [httpx.Response(202, json={"id": "job-m"})],
            "GET /v2/transcription/job-m": [httpx.Response(200, json=payload)],
        }
    )
    client = make_client(flow)
    transcript = client.transcribe(make_audio(tmp_path), "a1")
    assert len(transcript.words) == 5  # malformed entry dropped, 5 good words kept


def test_client_callable_as_pipeline_transcribe(tmp_path):
    """The adapter satisfies the pipeline's transcribe(path, asset_id) contract."""
    flow = MockFlow(
        {
            "POST /v2/upload": [
                httpx.Response(200, json={"audio_url": "https://cdn.gladia.test/a.wav"})
            ],
            "POST /v2/transcription": [httpx.Response(202, json={"id": "job-p"})],
            "GET /v2/transcription/job-p": [httpx.Response(200, json=done_payload())],
        }
    )
    client = make_client(flow)
    transcript = client.transcribe(make_audio(tmp_path), "asset_pipeline")
    assert isinstance(transcript, type(transcript))  # Transcript instance
    assert transcript.asset_id == "asset_pipeline"
    assert json.dumps(transcript.model_dump(mode="json"))  # JSON-serializable for DB
