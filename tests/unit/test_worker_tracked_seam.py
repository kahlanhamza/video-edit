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
