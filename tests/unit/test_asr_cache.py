"""Content-addressed ASR cache — the never-upload-twice law (owner 2026-11-23).

- miss: upload + create happen ONCE, and the id is recorded BEFORE polling
- retry after a poll crash/timeout: ADOPTS the recorded id — no second upload
- local hit (done + stored payload): ZERO network calls, any asset id
- provider-side failure: the row burns to `error` -> the next attempt starts
  a fresh upload instead of polling a dead id
- digest: same bytes -> same key (content addressing), different bytes differ
"""

from __future__ import annotations

from pathlib import Path

import pytest
from director_os.core.db import (
    ASRCacheRow,
    init_models,
    make_engine,
    make_session_factory,
)
from director_os.perception.asr_cache import GLADIA_PROVIDER, CachedTranscriber, sha256_file
from director_os.perception.gladia import GladiaError
from director_os.perception.transcript import Transcript, WordTiming


@pytest.fixture()
def sessions(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/asr_cache.db")
    init_models(engine)
    return make_session_factory(engine)


def make_media(tmp_path: Path, payload: bytes = b"bodycam-bytes-v1") -> Path:
    p = tmp_path / "media.mp4"
    p.write_bytes(payload)
    return p


class FakeGladia:
    """Scriptable client surface — counts every network-ish call."""

    def __init__(self, *, fail_fetch: str | None = None):
        self.uploads = 0
        self.creates = 0
        self.fetches = 0
        self.fail_fetch = fail_fetch  # substring of the error to raise

    def upload(self, path: Path) -> str:
        self.uploads += 1
        return "https://cdn.gladia.test/x"

    def create_job(self, audio_url: str, language: str | None = None) -> str:
        self.creates += 1
        return "job-1"

    def fetch_result(self, transcription_id: str) -> dict:
        self.fetches += 1
        if self.fail_fetch:
            raise GladiaError(self.fail_fetch)
        return {"id": transcription_id, "status": "done", "result": {}}

    def normalize(self, payload: dict, asset_id: str) -> Transcript:
        return Transcript(
            asset_id=asset_id,
            language="en",
            duration_s=2.3,
            words=[
                WordTiming(text="units", start_s=0.1, end_s=0.5, speaker="1"),
                WordTiming(text="arriving", start_s=0.6, end_s=1.2, speaker="1"),
            ],
            provider="gladia",
        )


def _row(sessions, digest: str) -> ASRCacheRow | None:
    with sessions() as s:
        return s.get(ASRCacheRow, (GLADIA_PROVIDER, digest))


def test_miss_uploads_once_then_persists_done(sessions, tmp_path):
    media = make_media(tmp_path)
    fake = FakeGladia()
    t = CachedTranscriber(fake, sessions).transcribe(media, "asset-1")

    assert fake.uploads == 1 and fake.creates == 1 and fake.fetches == 1
    assert t.asset_id == "asset-1" and t.words[0].text == "units"
    row = _row(sessions, sha256_file(media))
    assert row is not None and row.status == "done"
    assert row.external_id == "job-1"
    assert row.transcript_json and row.transcript_json["words"]


def test_local_hit_is_zero_network_and_rebinds_asset(sessions, tmp_path):
    media = make_media(tmp_path)
    CachedTranscriber(FakeGladia(), sessions).transcribe(media, "asset-1")

    # a LATER run (fresh client, zero calls so far) reuses the stored payload
    fresh = FakeGladia()
    t2 = CachedTranscriber(fresh, sessions).transcribe(media, "asset-2")
    assert fresh.uploads == 0 and fresh.creates == 0 and fresh.fetches == 0
    assert t2.asset_id == "asset-2"  # rebound, not stale
    assert t2.words[0].text == "units"


def test_processing_row_adopts_the_recorded_id(sessions, tmp_path):
    media = make_media(tmp_path)
    digest = sha256_file(media)
    with sessions() as s:
        s.add(
            ASRCacheRow(
                provider=GLADIA_PROVIDER,
                content_sha256=digest,
                external_id="job-9",
                status="processing",
            )
        )
        s.commit()

    fake = FakeGladia()
    t = CachedTranscriber(fake, sessions).transcribe(media, "asset-1")
    assert fake.uploads == 0 and fake.creates == 0  # adoption, never re-upload
    assert fake.fetches == 1
    assert t.asset_id == "asset-1"
    assert _row(sessions, digest).status == "done"  # type: ignore[union-attr]


def test_provider_failure_burns_row_then_fresh_upload(sessions, tmp_path):
    media = make_media(tmp_path)
    digest = sha256_file(media)
    with sessions() as s:
        s.add(
            ASRCacheRow(
                provider=GLADIA_PROVIDER,
                content_sha256=digest,
                external_id="job-9",
                status="processing",
            )
        )
        s.commit()

    dead = FakeGladia(fail_fetch="gladia job job-9 failed: status=error")
    with pytest.raises(GladiaError):
        CachedTranscriber(dead, sessions).transcribe(media, "asset-1")
    assert _row(sessions, digest).status == "error"  # type: ignore[union-attr]

    # next attempt: a fresh upload is allowed (the dead id is not polled forever)
    fresh = FakeGladia()
    CachedTranscriber(fresh, sessions).transcribe(media, "asset-1")
    assert fresh.uploads == 1


def test_poll_timeout_keeps_processing_for_adoption(sessions, tmp_path):
    media = make_media(tmp_path)
    digest = sha256_file(media)
    slow = FakeGladia(fail_fetch="gladia job job-1 timed out after 600s (last status: processing)")
    with pytest.raises(GladiaError):
        CachedTranscriber(slow, sessions).transcribe(media, "asset-1")

    row = _row(sessions, digest)
    assert row is not None
    assert row.status == "processing" and row.external_id  # id kept for adoption
    assert slow.uploads == 1  # but the upload happened only ONCE

    adopt = FakeGladia()
    CachedTranscriber(adopt, sessions).transcribe(media, "asset-1")
    assert adopt.uploads == 0 and adopt.fetches == 1  # adopts, no re-upload


def test_digest_is_content_addressed(tmp_path):
    a = tmp_path / "a.mp4"
    a.write_bytes(b"same-bytes")
    b = tmp_path / "copy.mp4"
    b.write_bytes(a.read_bytes())
    c = tmp_path / "c.mp4"
    c.write_bytes(b"different-bytes")

    assert sha256_file(a) == sha256_file(b)
    assert sha256_file(a) != sha256_file(c)
    assert len(sha256_file(a)) == 64
