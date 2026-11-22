"""Credential vault + settings overlay (ADR-0012) — hermetic unit tests.

Pins: whitelist fail-closed, version counter, masked reads, the overlay
taking precedence over env inside get_settings(), and the worker seam
(make_deps) re-reading saved credentials per job without restart.
"""

from __future__ import annotations

import base64

import pytest
from director_os.core.config import get_settings, set_settings_overrides
from director_os.core.credentials import (
    ALLOWED_CREDENTIAL_KEYS,
    VERSION_KEY,
    CredentialRow,
    SettingsMetaRow,
    apply_credentials_to_settings,
    delete_credential,
    encrypt_value,
    is_secret_key,
    load_credentials,
    mask_secret,
    read_version,
    upsert_credentials,
)
from director_os.core.db import make_engine, make_session_factory


@pytest.fixture()
def db(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/vault.db")
    CredentialRow.__table__.create(engine)
    SettingsMetaRow.__table__.create(engine)
    return engine


def test_whitelist_rejects_unknown_keys(db):
    with make_session_factory(db)() as s:
        with pytest.raises(KeyError):
            upsert_credentials(s, {"DATABASE_URL": "postgres://evil"})
        assert read_version(s) == 0
        assert load_credentials(s) == {}


def test_upsert_roundtrip_and_version(db):
    with make_session_factory(db)() as s:
        assert upsert_credentials(s, {"GLADIA_API_KEY": "glad_live_abc123"}) == 1
        assert (
            upsert_credentials(
                s, {"GLADIA_API_KEY": "glad_live_rotated", "LLM_MODEL": "muse-ember-1.0"}
            )
            == 2
        )
        assert load_credentials(s) == {
            "GLADIA_API_KEY": "glad_live_rotated",
            "LLM_MODEL": "muse-ember-1.0",
        }
        assert delete_credential(s, "GLADIA_API_KEY") == 3
        assert load_credentials(s) == {"LLM_MODEL": "muse-ember-1.0"}
        # deleting a missing key is a no-op that still succeeds
        assert delete_credential(s, "GLADIA_API_KEY") == 3


def test_overlay_beats_env_in_get_settings(db, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "env-model")
    monkeypatch.delenv("GLADIA_API_KEY", raising=False)
    with make_session_factory(db)() as s:
        upsert_credentials(s, {"LLM_MODEL": "studio-model", "GLADIA_API_KEY": "glad_key"})
    apply_credentials_to_settings(db)
    s = get_settings()
    assert s.llm_model == "studio-model"
    assert s.gladia_api_key == "glad_key"
    set_settings_overrides({})


def test_secret_masking_rules():
    assert is_secret_key("GLADIA_API_KEY")
    assert is_secret_key("MODAL_TOKEN_SECRET")
    assert is_secret_key("YOUTUBE_REFRESH_TOKEN")
    assert not is_secret_key("LLM_MODEL")
    assert not is_secret_key("LLM_BASE_URL")
    assert mask_secret("") == ""
    assert mask_secret("abc") == "••••"
    assert mask_secret("glad_live_abc123") == "••••c123"
    assert "glad_live" not in mask_secret("glad_live_abc123")


def test_plaintext_fallback_without_secret_key(monkeypatch):
    monkeypatch.delenv("SETTINGS_SECRET_KEY", raising=False)
    stored, encrypted = encrypt_value("secret-value")
    assert stored == "secret-value" and encrypted is False


def test_encryption_path_with_fake_fernet(monkeypatch):
    """SETTINGS_SECRET_KEY set + a fernet-compatible module -> stored value is
    encrypted (roundtrip decrypts) and an undecryptable row is skipped, never
    crashes the loader."""

    class _FakeFernet:
        def encrypt(self, data: bytes) -> bytes:
            return b"fernet:" + base64.b64encode(data)

        def decrypt(self, data: bytes) -> bytes:
            assert data.startswith(b"fernet:"), "not our token"
            return base64.b64decode(data[7:])

    monkeypatch.setenv("SETTINGS_SECRET_KEY", "unit-test-secret")
    import director_os.core.credentials as cred

    monkeypatch.setattr(cred, "_fernet", lambda: _FakeFernet())
    stored, encrypted = encrypt_value("secret-value")
    assert encrypted is True and stored.startswith("fernet:")
    assert cred.decrypt_value(stored, encrypted) == "secret-value"

    engine = make_engine("sqlite:///:memory:")
    CredentialRow.__table__.create(engine)
    SettingsMetaRow.__table__.create(engine)
    with make_session_factory(engine)() as s:
        s.add(CredentialRow(key="GLADIA_API_KEY", value=stored, encrypted=True))
        s.add(CredentialRow(key="LLM_MODEL", value="muse-ember-1.0", encrypted=False))
        s.commit()
        creds = load_credentials(s)
    assert creds["GLADIA_API_KEY"] == "secret-value"
    assert creds["LLM_MODEL"] == "muse-ember-1.0"
    # a row encrypted under a DIFFERENT key must be skipped, not fatal
    with make_session_factory(engine)() as s:
        s.add(CredentialRow(key="VISION_API_KEY", value="fernet:garbage", encrypted=True))
        s.commit()
        creds = load_credentials(s)
    assert "VISION_API_KEY" not in creds


def test_whitelist_covers_provider_catalog_fields():
    """Every field the Settings UI renders must be storable."""
    from director_os.api.main import _PROVIDER_SPECS

    ui_keys = {f["key"] for spec in _PROVIDER_SPECS for f in spec["fields"]}
    assert ui_keys, "catalog must not be empty"
    missing = ui_keys - set(ALLOWED_CREDENTIAL_KEYS)
    assert not missing, f"UI fields missing from the vault whitelist: {sorted(missing)}"


def test_worker_seam_reads_saved_credentials_per_job(tmp_path, monkeypatch):
    """make_deps(settings=None) refreshes the overlay from the DB on EVERY
    job — a key saved in the Studio takes effect without a worker restart."""
    import director_os.worker.pipeline as pipeline

    monkeypatch.setenv("LLM_MODEL", "env-model")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    url = f"sqlite:///{tmp_path}/worker.db"

    # job 1: no credentials saved yet -> env wins (tables get created here)
    deps1 = pipeline.make_deps(url, transcribe=None)
    assert deps1.settings.llm_model == "env-model"

    # the Studio saves a key into the SAME database the worker reads
    with make_session_factory(make_engine(url))() as s:
        upsert_credentials(s, {"LLM_MODEL": "studio-model-per-job"})

    # job 2 (same process, no restart): the saved key wins
    deps2 = pipeline.make_deps(url, transcribe=None)
    assert deps2.settings.llm_model == "studio-model-per-job"


def test_version_meta_defaults_zero(db):
    with make_session_factory(db)() as s:
        assert read_version(s) == 0
        assert s.get(SettingsMetaRow, VERSION_KEY) is None
