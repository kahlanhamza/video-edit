"""Runtime settings — 12-factor, env-driven (ADR: config is data, not code)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

LOCAL_STORAGE = "local"
S3_STORAGE = "s3"


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


def _load() -> Settings:
    output_dir = Path(os.getenv("OUTPUT_DIR", "./output")).resolve()
    return Settings(
        database_url=os.getenv("DATABASE_URL", "sqlite:///./output/directoros.db"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        storage_backend=os.getenv("STORAGE_BACKEND", LOCAL_STORAGE),
        output_dir=output_dir,
        ingest_dir=output_dir / "ingest",
        segments_dir=output_dir / "segments",
        target_lufs=float(os.getenv("TARGET_LUFS", "-16.0")),
        sample_rate=int(os.getenv("SAMPLE_RATE", "48000")),
        channels=int(os.getenv("CHANNELS", "2")),
        fps=int(os.getenv("FPS", "30")),
        res_w=int(os.getenv("RES_W", "1920")),
        res_h=int(os.getenv("RES_H", "1080")),
        contact_email=os.getenv("CONTACT_EMAIL", "ops@directoros.example"),
        r2_endpoint=os.getenv("R2_ENDPOINT", ""),
        r2_access_key_id=os.getenv("R2_ACCESS_KEY_ID", ""),
        r2_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", ""),
        r2_bucket=os.getenv("R2_BUCKET", "directoros"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return _load()
