"""S3ObjectStore._ensure_bucket — the belt-and-suspenders bucket auto-create.

Guarantees for the one-command compose setup: the very first write must
succeed even when no pre-creation step ran (fresh machine, wiped volume),
and the check must be cheap (head_bucket once, then cached) and never
mask unexpected S3 errors.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from botocore.exceptions import ClientError
from director_os.core.config import get_settings
from director_os.core.storage import S3ObjectStore


class _FakeS3:
    def __init__(self, head_error: str | None):
        self.calls: list[tuple[str, Any]] = []
        self._head_error = head_error

    def head_bucket(self, **kw: Any) -> None:
        self.calls.append(("head_bucket", kw.get("Bucket")))
        if self._head_error:
            raise ClientError(
                {"Error": {"Code": self._head_error, "Message": "boom"}},
                "HeadBucket",
            )

    def create_bucket(self, **kw: Any) -> None:
        self.calls.append(("create_bucket", kw.get("Bucket")))

    def upload_file(self, *a: Any, **kw: Any) -> None:
        self.calls.append(("upload_file", a or kw))

    def download_file(self, *a: Any, **kw: Any) -> None:
        self.calls.append(("download_file", a or kw))


def _store(error_code: str | None) -> tuple[S3ObjectStore, _FakeS3]:
    client = _FakeS3(error_code)
    store = S3ObjectStore(
        dataclasses.replace(
            get_settings(),
            storage_backend="s3",
            r2_endpoint="http://minio:9000",
            r2_access_key_id="test",
            r2_secret_access_key="test-secret",
            r2_bucket="directoros",
        )
    )
    store._client = lambda: client  # type: ignore[method-assign]
    return store, client


def test_existing_bucket_no_create() -> None:
    store, client = _store(None)
    store.put_file("a/b.txt", "/dev/null")
    ops = [op for op, _ in client.calls]
    assert ops == ["head_bucket", "upload_file"]


@pytest.mark.parametrize("code", ["404", "NoSuchBucket"])
def test_missing_bucket_is_created_then_cached(code: str) -> None:
    store, client = _store(code)
    store.put_file("a/b.txt", "/dev/null")
    ops = [op for op, _ in client.calls]
    assert ops == ["head_bucket", "create_bucket", "upload_file"]
    # second write: check is cached — no repeated head/create chatter
    store.put_file("c/d.txt", "/dev/null")
    ops2 = [op for op, _ in client.calls]
    assert ops2 == ["head_bucket", "create_bucket", "upload_file", "upload_file"]


def test_unexpected_s3_error_is_never_masked() -> None:
    store, client = _store("403")
    with pytest.raises(ClientError):
        store.put_file("a/b.txt", "/dev/null")
    assert [op for op, _ in client.calls] == ["head_bucket"]


def test_get_file_also_ensures_bucket() -> None:
    store, client = _store("404")
    store.get_file("a/b.txt", "/tmp/whatever")
    ops = [op for op, _ in client.calls]
    assert ops[0] == "head_bucket"
    assert "create_bucket" in ops
