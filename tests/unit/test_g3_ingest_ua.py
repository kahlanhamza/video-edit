"""Guardrail 3 (PLAN §9.3): every cloud ingest sends a descriptive User-Agent."""

from __future__ import annotations

from director_os.core.http import DEFAULT_UA, build_headers, download


def test_default_headers_carry_directoros_identity():
    headers = build_headers(contact_email="ops@directoros.example")
    assert "DirectorOS/" in headers["User-Agent"]
    assert "ops@directoros.example" in headers["User-Agent"]
    assert DEFAULT_UA.startswith("DirectorOS/")


def test_caller_cannot_spoof_or_drop_the_ua():
    headers = build_headers(
        "ops@directoros.example", {"User-Agent": "curl/8.0", "Cookie": "session=1"}
    )
    assert "DirectorOS/" in headers["User-Agent"], "UA override must be ignored"
    assert headers["Cookie"] == "session=1", "extra headers still honored"


def test_download_forces_identity_headers(monkeypatch, tmp_path):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def iter_bytes(self, chunk_size):
            yield b"data"

    from contextlib import contextmanager

    @contextmanager
    def fake_stream(method, url, headers=None, **kwargs):
        captured["headers"] = headers
        yield FakeResponse()

    monkeypatch.setattr("director_os.core.http.httpx.stream", fake_stream)
    dest = download("https://example.org/clip.webm", tmp_path / "clip.webm")
    assert dest.exists()
    assert captured["headers"]["User-Agent"].startswith("DirectorOS/")
