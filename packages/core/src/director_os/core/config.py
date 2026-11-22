"""Runtime settings — 12-factor, env-driven (ADR: config is data, not code)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

LOCAL_STORAGE = "local"
S3_STORAGE = "s3"

# Runtime credential overlay (Studio Settings -> DB -> here, ADR-0012):
# keys saved in the Settings UI override the process env inside get_settings()
# WITHOUT a restart. Set/refreshed by core.credentials.apply_credentials_to_settings().
_OVERRIDES: dict[str, str] = {}


def set_settings_overrides(overrides: dict[str, str]) -> None:
    """Replace the overlay (whitelist filtering is the caller's job) and
    invalidate the cached Settings so the next get_settings() sees it."""
    global _OVERRIDES
    _OVERRIDES = dict(overrides)
    get_settings.cache_clear()


def _env(name: str, default: str = "") -> str:
    if name in _OVERRIDES:
        return _OVERRIDES[name]
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    database_url: str
    redis_url: str
    storage_backend: str
    output_dir: Path
    ingest_dir: Path
    segments_dir: Path
    # --- the audio law (PLAN guardrail 1, ADR-0005)
    target_lufs: float
    sample_rate: int
    channels: int
    # --- video defaults
    fps: int
    res_w: int
    res_h: int
    # --- ingest identity (PLAN guardrail 3)
    contact_email: str
    # --- object storage (R2 / MinIO)
    r2_endpoint: str
    r2_access_key_id: str
    r2_secret_access_key: str
    r2_bucket: str
    # --- external services (M2)
    gladia_api_key: str
    # --- agent runtime LLM (M2 close-out)
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    # --- Meta Model API cost guard (M2: provider = meta)
    llm_provider: str
    llm_budget_usd: float
    llm_spend_ledger: str
    llm_allow_training_on_content: bool
    # --- capacity routing (comma-separated model chains per backend)
    llm_fallback_models: str
    gemini_fallback_models: str
    # --- OpenRouter secondary tier (openai-compat; :free models)
    openrouter_api_key: str
    openrouter_base_url: str
    openrouter_model: str
    openrouter_fallback_models: str
    # --- vision LLM (M4)
    vision_api_key: str
    vision_provider: str
    vision_model: str
    vision_fallback_models: str
    # --- compute backend (M4: local ONNX default / Kaggle free GPU / Modal optional)
    compute_backend: str
    kaggle_username: str
    kaggle_key: str
    # --- Modal GPU (M4, dormant until a card exists)
    modal_token_id: str
    modal_token_secret: str
    # --- tts providers (M2-M5: edge default $0, elevenlabs optional studio grade)
    tts_provider: str
    elevenlabs_api_key: str
    elevenlabs_voice_id: str
    elevenlabs_model_id: str


def _load() -> Settings:
    output_dir = Path(os.getenv("OUTPUT_DIR", "./output")).resolve()
    return Settings(
        database_url=os.getenv("DATABASE_URL", "sqlite:///./output/directoros.db"),
        redis_url=_env("REDIS_URL", "redis://localhost:6379/0"),
        storage_backend=_env("STORAGE_BACKEND", LOCAL_STORAGE),
        output_dir=output_dir,
        ingest_dir=output_dir / "ingest",
        segments_dir=output_dir / "segments",
        target_lufs=float(os.getenv("TARGET_LUFS", "-16.0")),
        sample_rate=int(os.getenv("SAMPLE_RATE", "48000")),
        channels=int(os.getenv("CHANNELS", "2")),
        fps=int(os.getenv("FPS", "30")),
        res_w=int(os.getenv("RES_W", "1920")),
        res_h=int(os.getenv("RES_H", "1080")),
        contact_email=_env("CONTACT_EMAIL", "ops@directoros.example"),
        r2_endpoint=_env("R2_ENDPOINT", ""),
        r2_access_key_id=_env("R2_ACCESS_KEY_ID", ""),
        r2_secret_access_key=_env("R2_SECRET_ACCESS_KEY", ""),
        r2_bucket=_env("R2_BUCKET", "directoros"),
        gladia_api_key=_env("GLADIA_API_KEY", ""),
        llm_api_key=_env("LLM_API_KEY", ""),
        llm_base_url=_env("LLM_BASE_URL", ""),
        llm_model=_env("LLM_MODEL", ""),
        llm_provider=_env("LLM_PROVIDER", ""),
        llm_budget_usd=float(_env("LLM_BUDGET_USD", "15.0") or "15.0"),
        llm_spend_ledger=_env("LLM_SPEND_LEDGER", ""),
        llm_allow_training_on_content=(
            _env("LLM_ALLOW_TRAINING_ON_CONTENT", "").lower() in {"1", "true", "yes"}
        ),
        llm_fallback_models=_env("LLM_FALLBACK_MODELS", ""),
        gemini_fallback_models=_env("GEMINI_FALLBACK_MODELS", ""),
        openrouter_api_key=_env("OPENROUTER_API_KEY", ""),
        openrouter_base_url=_env("OPENROUTER_BASE_URL", ""),
        openrouter_model=_env("OPENROUTER_MODEL", ""),
        openrouter_fallback_models=os.getenv("OPENROUTER_FALLBACK_MODELS", ""),
        vision_api_key=_env("VISION_API_KEY", ""),
        vision_provider=os.getenv("VISION_PROVIDER", ""),
        vision_model=_env("VISION_MODEL", "gemini-3.5-flash-lite"),
        vision_fallback_models=os.getenv("VISION_FALLBACK_MODELS", "gemini-3.1-flash-lite"),
        compute_backend=_env("COMPUTE_BACKEND", "local").strip().lower(),
        kaggle_username=_env("KAGGLE_USERNAME", ""),
        kaggle_key=_env("KAGGLE_KEY", ""),
        modal_token_id=_env("MODAL_TOKEN_ID", ""),
        modal_token_secret=_env("MODAL_TOKEN_SECRET", ""),
        tts_provider=os.getenv("TTS_PROVIDER", "edge").strip().lower(),
        elevenlabs_api_key=_env("ELEVENLABS_API_KEY", ""),
        elevenlabs_voice_id=_env("ELEVENLABS_VOICE_ID", ""),
        elevenlabs_model_id=os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return _load()
