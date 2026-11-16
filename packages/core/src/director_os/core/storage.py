"""Object storage abstraction: local FS backend (dev/CI) + S3/R2 backend (prod).

boto3 is imported lazily so the core wheel stays light and CI never touches S3.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol

from director_os.core.config import S3_STORAGE, Settings


class ObjectStore(Protocol):
    def put_file(self, key: str, src_path: str | Path) -> str: ...
    def get_file(self, key: str, dest_path: str | Path) -> Path: ...
    def exists(self, key: str) -> bool: ...


class LocalObjectStore:
    """Filesystem-backed store; keys are relative paths under a base dir."""

    def __init__(self, base_dir: str | Path):
        self.base = Path(base_dir).resolve()
        self.base.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        p = (self.base / key).resolve()
        if not str(p).startswith(str(self.base)):
            raise ValueError(f"illegal storage key: {key}")
        return p

    def put_file(self, key: str, src_path: str | Path) -> str:
        dest = self._resolve(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest)
        return key

    def get_file(self, key: str, dest_path: str | Path) -> Path:
        src = self._resolve(key)
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return dest

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()


class S3ObjectStore:
    """S3-compatible store (Cloudflare R2 / MinIO / AWS). boto3 loaded lazily."""

    def __init__(self, settings: Settings):
        if not settings.r2_endpoint:
            raise ValueError("STORAGE_BACKEND=s3 requires R2_ENDPOINT")
        self.settings = settings

    def _client(self):
        import boto3  # lazy: keeps boto3 out of the default install surface

        return boto3.client(
            "s3",
            endpoint_url=self.settings.r2_endpoint,
            aws_access_key_id=self.settings.r2_access_key_id,
            aws_secret_access_key=self.settings.r2_secret_access_key,
        )

    def put_file(self, key: str, src_path: str | Path) -> str:
        self._client().upload_file(str(src_path), self.settings.r2_bucket, key)
        return key

    def get_file(self, key: str, dest_path: str | Path) -> Path:
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._client().download_file(self.settings.r2_bucket, key, str(dest))
        return dest

    def exists(self, key: str) -> bool:
        try:
            self._client().head_object(Bucket=self.settings.r2_bucket, Key=key)
            return True
        except Exception:
            return False


def store_from_env(
    settings: Settings | None = None, base_dir: str | Path | None = None
) -> ObjectStore:
    s = settings or get_settings_safe()
    if s.storage_backend == S3_STORAGE:
        return S3ObjectStore(s)
    base = Path(base_dir) if base_dir else s.output_dir / "store"
    return LocalObjectStore(base)


def get_settings_safe() -> Settings:
    from director_os.core.config import get_settings

    return get_settings()
