"""Worker seam — REDACT_MODE env flow + bake guardrails (issue #14 closure)."""

from director_os.worker.pipeline import _tracked_redacted_source, make_deps


def test_make_deps_reads_redact_mode_env(monkeypatch, tmp_path):
    monkeypatch.setenv("REDACT_MODE", "tracked")
    deps = make_deps(f"sqlite:///{tmp_path}/a.db")
    assert deps.redact_backend == "tracked"


def test_make_deps_explicit_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("REDACT_MODE", "tracked")
    deps = make_deps(f"sqlite:///{tmp_path}/b.db", redact_backend="yunet")
    assert deps.redact_backend == "yunet"


def test_make_deps_unset_env_is_hermetic(monkeypatch, tmp_path):
    monkeypatch.delenv("REDACT_MODE", raising=False)
    deps = make_deps(f"sqlite:///{tmp_path}/c.db")
    assert deps.redact_backend is None  # CI stays model-free by default


def test_baked_source_none_without_payload(tmp_path):
    deps = make_deps(f"sqlite:///{tmp_path}/d.db", redact_backend="tracked")
    out = _tracked_redacted_source(deps, "p1", tmp_path / "src.mp4", "asset1", tmp_path)
    assert out is None  # no persisted plan -> raw source stays in play


def test_baked_source_none_when_mode_not_tracked(tmp_path):
    deps = make_deps(f"sqlite:///{tmp_path}/e.db", redact_backend=None)
    out = _tracked_redacted_source(deps, "p1", tmp_path / "src.mp4", "asset1", tmp_path)
    assert out is None


def test_hydrate_blur_regions_round_trips_payload_dicts():
    """The tracked payload is persisted as JSON dicts; stage_redaction must
    re-hydrate them to BlurRegion models before the shared model_dump build
    (live full-tape run 3: 'dict' object has no attribute 'model_dump')."""
    from director_os.core.models import BlurRegion
    from director_os.worker.pipeline import _hydrate_blur_regions

    model = BlurRegion.model_validate(
        {"bbox": {"x": 1, "y": 2, "w": 3, "h": 4}, "mode": "tracked", "track_id": "face_1"}
    )
    payload = [model.model_dump(mode="json"), {"bbox": {"x": 5, "y": 6, "w": 7, "h": 8}}]
    out = _hydrate_blur_regions(payload)
    assert all(isinstance(r, BlurRegion) for r in out)
    twice = _hydrate_blur_regions(out)
    assert [r.model_dump(mode="json") for r in out] == [r.model_dump(mode="json") for r in twice]
    assert out[1].mode == "static"
