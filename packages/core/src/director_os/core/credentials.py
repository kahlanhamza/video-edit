"""Credential vault — Studio-managed keys that override .env at runtime.

Big-tech provider-settings flow (ADR-0012): the Settings UI saves API keys
INTO the database (encrypted at rest when `SETTINGS_SECRET_KEY` is set and
`cryptography` is importable — plaintext by default, the same threat model
as the .env file itself). `set_settings_overrides()` merges them over the
process env inside `get_settings()`, so the LLM chain, ASR, TTS and compute
backends pick up saved keys WITHOUT a restart: the gateway refreshes on
startup + every save, the worker fleet refreshes in `make_deps` per job.

Laws:
- every key is on a strict whitelist (ALLOWED_CREDENTIAL_KEYS) — unknown
  keys are rejected, not stored;
- secrets are NEVER returned by the API — masked hints only;
- version counter (`credentials_version` meta row) lets long-lived worker
  processes detect changes cheaply (one SELECT per job).
"""

from __future__ import annotations

import base64
import os
from datetime import UTC, datetime
from typing import Any

from director_os.core.db import Base, _utcnow, new_id
from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

VERSION_KEY = "credentials_version"

# --- the whitelist: every key the Studio Settings UI may save -------------
# (key = the env-var name it overrides; secrets are masked on read)
ALLOWED_CREDENTIAL_KEYS: tuple[str, ...] = (
    # LLM (primary provider: openai-compat / gemini / meta)
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_PROVIDER",
    "LLM_BUDGET_USD",
    "LLM_FALLBACK_MODELS",
    "GEMINI_FALLBACK_MODELS",
    "LLM_ALLOW_TRAINING_ON_CONTENT",
    # vision LLM (frame understanding)
    "VISION_API_KEY",
    "VISION_BASE_URL",
    "VISION_MODEL",
    # ASR
    "GLADIA_API_KEY",
    # secondary openai-compat tier
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_MODEL",
    # compute backends
    "COMPUTE_BACKEND",
    "KAGGLE_USERNAME",
    "KAGGLE_KEY",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    # TTS
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_VOICE_ID",
    # YouTube publish
    "YOUTUBE_CLIENT_ID",
    "YOUTUBE_CLIENT_SECRET",
    "YOUTUBE_REFRESH_TOKEN",
    # ingest identity
    "CONTACT_EMAIL",
)

SECRET_MARKERS = ("API_KEY", "APIKEY", "_KEY", "SECRET", "TOKEN")


def is_secret_key(key: str) -> bool:
    """True when the value must be masked in API responses."""
    return any(marker in key for marker in SECRET_MARKERS)


class CredentialRow(Base):
    __tablename__ = "credentials"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    encrypted: Mapped[bool] = mapped_column(default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class SettingsMetaRow(Base):
    __tablename__ = "settings_meta"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(String(100), default="0")


# --- at-rest encryption (optional, zero-dep default) -----------------------


class _EncryptionUnavailable(Exception):
    pass


def _fernet():
    secret = os.getenv("SETTINGS_SECRET_KEY", "").strip()
    if not secret:
        raise _EncryptionUnavailable("SETTINGS_SECRET_KEY not set")
    try:
        from cryptography.fernet import Fernet
    except ImportError as e:  # pragma: no cover - exercised only with dep absent
        raise _EncryptionUnavailable("cryptography package not installed") from e
    digest = __import__("hashlib").sha256(secret.encode()).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt_value(value: str) -> tuple[str, bool]:
    """Return (stored_value, encrypted_flag). Plaintext when no key is set."""
    try:
        return _fernet().encrypt(value.encode()).decode(), True
    except _EncryptionUnavailable:
        return value, False


def decrypt_value(stored: str, encrypted: bool) -> str:
    if not encrypted:
        return stored
    try:
        return _fernet().decrypt(stored.encode()).decode()
    except Exception as e:
        raise ValueError(f"cannot decrypt credential (SETTINGS_SECRET_KEY changed?): {e}") from e


def encryption_active() -> bool:
    try:
        _fernet()
        return True
    except _EncryptionUnavailable:
        return False


# --- store operations (plain SQLAlchemy, session passed in) ----------------


def read_version(session: Any) -> int:
    row = session.get(SettingsMetaRow, VERSION_KEY)
    return int(row.value) if row else 0


def _bump_version(session: Any) -> None:
    row = session.get(SettingsMetaRow, VERSION_KEY)
    if row is None:
        session.add(SettingsMetaRow(key=VERSION_KEY, value="1"))
    else:
        row.value = str(int(row.value) + 1)


def load_credentials(session: Any) -> dict[str, str]:
    """All stored credentials, decrypted. Used by the overlay loader."""
    out: dict[str, str] = {}
    for row in session.query(CredentialRow).all():
        try:
            out[row.key] = decrypt_value(row.value, row.encrypted)
        except ValueError:
            # undecryptable rows are skipped, never crash the pipeline
            continue
    return out


def upsert_credentials(session: Any, entries: dict[str, str]) -> int:
    """Whitelist-checked upsert; returns the new version."""
    unknown = [k for k in entries if k not in ALLOWED_CREDENTIAL_KEYS]
    if unknown:
        raise KeyError(
            f"unknown credential keys: {unknown} (allowed: {len(ALLOWED_CREDENTIAL_KEYS)})"
        )
    for key, value in entries.items():
        value = value.strip()
        if not value:
            continue
        stored, encrypted = encrypt_value(value)
        row = session.get(CredentialRow, key)
        if row is None:
            session.add(
                CredentialRow(
                    key=key,
                    value=stored,
                    encrypted=encrypted,
                    updated_at=datetime.now(UTC),
                )
            )
        else:
            row.value = stored
            row.encrypted = encrypted
            row.updated_at = datetime.now(UTC)
    if entries:
        _bump_version(session)
    session.commit()
    return read_version(session)


def delete_credential(session: Any, key: str) -> int:
    if key not in ALLOWED_CREDENTIAL_KEYS:
        raise KeyError(f"unknown credential key: {key}")
    row = session.get(CredentialRow, key)
    if row is not None:
        session.delete(row)
        _bump_version(session)
        session.commit()
    return read_version(session)


def mask_secret(value: str) -> str:
    """Stable masked hint: keep the last 4 chars for identity comparison."""
    v = value.strip()
    if not v:
        return ""
    if len(v) <= 6:
        return "••••"
    return f"••••{v[-4:]}"


# --- overlay into get_settings() -------------------------------------------


def apply_credentials_to_settings(engine: Any) -> int:
    """Load DB credentials into the settings overlay (idempotent).

    Returns the credentials version now in effect. Call sites: gateway
    startup + every save, worker `make_deps` per job (version-guarded).
    """
    from director_os.core.config import set_settings_overrides
    from sqlalchemy.orm import Session

    with Session(engine) as s:
        version = read_version(s)
        creds = load_credentials(s)
    set_settings_overrides(creds)
    return version


def new_credential_id() -> str:
    return new_id()
