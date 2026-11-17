"""Kaggle batch-compute client — free cloud GPU, zero SDK, pure httpx.

Kaggle's free tier ships a Tesla T4 (15.6 GB VRAM) per kernel session — no
payment method, no card verification, just an API token. DirectorOS uses it
as the `KaggleBatchGpuBackend` transport for heavy vision/audio jobs
(WhisperX, Demucs, YOLO sweeps) that would crawl on CPU.

Auth (2025+ tokens): `KGAT_*` keys ride `Authorization: Bearer ...`; the
username is discovered via `/api/v1/hello` (response `userName`) so the
`.env` only needs the token. Legacy 32-hex keys keep basic-auth
(`KAGGLE_USERNAME` + key).

Wire contract (verified against a live T4 run, 2026-11):
- POST   /api/v1/kernels/push          camelCase body, slug "user/slug"
- GET    /api/v1/kernels/status?userName=&kernelSlug=
         {"status": running|complete|error|...}
- GET    /api/v1/kernels/output?userName=&kernelSlug=
         {"logNullable": "<json str>", "files": [names]}
- GET    /api/v1/kernels/output/download/{owner}/{slug}/{file}

Everything the kernel writes to `/kaggle/working` lands in `files`; the
session log (stdout/stderr records) rides in `logNullable` and is parsed
into `KernelRunResult.log`.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from director_os.core.config import get_settings

DEFAULT_BASE_URL = "https://www.kaggle.com"
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_RUN_TIMEOUT_S = 30 * 60
DEFAULT_POLL_INTERVAL_S = 15.0
TERMINAL_STATUSES = {"complete", "error", "cancelAcknowledged"}


class KaggleError(RuntimeError):
    pass


class KaggleNotConfigured(KaggleError):
    """Raised when neither a KGAT token nor username+key is available."""


@dataclass(frozen=True)
class KernelRunResult:
    """Outcome of one push -> poll -> harvest cycle."""

    slug: str
    url: str
    status: str
    log: list[dict] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    out_dir: Path | None = None

    def stdout_text(self) -> str:
        """Concatenated stdout records (the log parse is tolerant)."""
        return "".join(
            str(rec.get("data", "")) for rec in self.log if rec.get("stream_name") == "stdout"
        )

    def result_json(self, marker: str = "DIRECTOROS_RESULT=") -> dict:
        """Parse `print("DIRECTOROS_RESULT=" + json.dumps(d))` from the log."""
        m = re.search(re.escape(marker) + r"(\{.*\})", self.stdout_text())
        if not m:
            raise KaggleError(f"kernel output missing {marker!r} marker")
        return json.loads(m.group(1))


class KaggleClient:
    """Minimal kernels API client. `transport` is injectable for hermetic tests."""

    def __init__(
        self,
        username: str = "",
        key: str = "",
        base_url: str = DEFAULT_BASE_URL,
        transport: httpx.BaseTransport | None = None,
    ):
        self._transport = transport
        self.base_url = base_url.rstrip("/")
        self.key = key.strip()
        self.username = username.strip()
        if not self.key:
            raise KaggleNotConfigured(
                "Kaggle not configured: set KAGGLE_KEY (KGAT_* token, "
                "or legacy KAGGLE_USERNAME+KEY)"
            )
        if not self.key.startswith("KGAT_") and not self.username:
            # legacy 32-hex keys authenticate per-user (basic auth) — fail fast
            # instead of discovering at first use; KGAT tokens can self-resolve
            raise KaggleNotConfigured("legacy Kaggle key requires KAGGLE_USERNAME")

    # ------------------------------------------------------------------ auth
    def _auth(self) -> tuple[dict[str, str], httpx.BasicAuth | None]:
        if self.key.startswith("KGAT_"):
            if not self.username:
                # KGAT tokens are account-scoped: /api/v1/hello reveals the user
                try:
                    with httpx.Client(transport=self._transport, timeout=DEFAULT_TIMEOUT_S) as c:
                        r = c.get(
                            f"{self.base_url}/api/v1/hello",
                            headers={"Authorization": f"Bearer {self.key}"},
                        )
                    if r.status_code == 200:
                        self.username = str(r.json().get("userName", ""))
                except httpx.HTTPError:
                    pass
                if not self.username:
                    raise KaggleNotConfigured(
                        "could not resolve username for KGAT token (/api/v1/hello failed) — "
                        "set KAGGLE_USERNAME explicitly"
                    )
            return {"Authorization": f"Bearer {self.key}"}, None
        return {}, httpx.BasicAuth(self.username, self.key)

    def _client(self) -> httpx.Client:
        headers, auth = self._auth()
        return httpx.Client(
            base_url=self.base_url,
            headers=headers,
            auth=auth,
            transport=self._transport,
            timeout=DEFAULT_TIMEOUT_S,
        )

    # -------------------------------------------------------------- endpoints
    def whoami(self) -> str:
        """Authenticated username (Bearer tokens learn it from /api/v1/hello)."""
        if not self.username:
            self._auth()
        return self.username

    def push_kernel(
        self,
        slug: str,
        title: str,
        script: str,
        *,
        gpu: bool = True,
        internet: bool = True,
        session_timeout_s: int | None = None,
    ) -> dict:
        """Push a script kernel (new version each push). camelCase body — the
        wire shape `kagglesdk` serializes for `/api/v1/kernels/push`."""
        if "/" not in slug:
            slug = f"{self.whoami()}/{slug}"
        if len(title) < 5:
            raise KaggleError("kernel title must be at least five characters")
        body: dict = {
            "slug": slug,
            "newTitle": title,
            "text": script,
            "language": "python",
            "kernelType": "script",
            "isPrivate": True,
            "enableGpu": gpu,
            "enableTpu": False,
            "enableInternet": internet,
            "datasetDataSources": [],
            "competitionDataSources": [],
            "kernelDataSources": [],
            "modelDataSources": [],
        }
        if session_timeout_s:
            body["sessionTimeoutSeconds"] = session_timeout_s
        with self._client() as c:
            r = c.post("/api/v1/kernels/push", json=body)
        if r.status_code != 200:
            raise KaggleError(f"kernels/push {r.status_code}: {r.text[:300]}")
        data = r.json()
        if data.get("error"):
            raise KaggleError(f"kernels/push rejected: {data['error']}")
        return data

    def kernel_status(self, slug: str) -> str:
        if "/" not in slug:
            slug = f"{self.whoami()}/{slug}"
        user, kslug = slug.split("/", 1)
        with self._client() as c:
            r = c.get("/api/v1/kernels/status", params={"userName": user, "kernelSlug": kslug})
        if r.status_code != 200:
            raise KaggleError(f"kernels/status {r.status_code}: {r.text[:200]}")
        return str(r.json().get("status", "unknown"))

    def kernel_output(self, slug: str) -> dict:
        """Session log + list of /kaggle/working output files."""
        if "/" not in slug:
            slug = f"{self.whoami()}/{slug}"
        user, kslug = slug.split("/", 1)
        with self._client() as c:
            r = c.get("/api/v1/kernels/output", params={"userName": user, "kernelSlug": kslug})
        if r.status_code != 200:
            raise KaggleError(f"kernels/output {r.status_code}: {r.text[:200]}")
        data = r.json()
        log: list[dict] = []
        raw_log = data.get("logNullable") or data.get("log") or ""
        if isinstance(raw_log, str) and raw_log.strip():
            try:
                parsed = json.loads(raw_log)
                if isinstance(parsed, list):
                    log = [rec for rec in parsed if isinstance(rec, dict)]
            except json.JSONDecodeError:
                log = [{"stream_name": "stdout", "data": raw_log[:10000]}]
        elif isinstance(raw_log, list):
            log = [rec for rec in raw_log if isinstance(rec, dict)]
        # The API returns file OBJECTS ({fileName/url} variants), not strings —
        # normalize so callers get stable names (a str(dict) repr 404s downloads).
        files: list[dict] = []
        for f in data.get("files", []):
            if isinstance(f, dict):
                name = str(f.get("fileName") or f.get("fileNameNullable") or "")
                url = str(f.get("url") or f.get("urlNullable") or "")
                if name:
                    files.append({"name": name, "url": url})
            elif isinstance(f, str):
                files.append({"name": f, "url": ""})
        return {"log": log, "files": files}

    def download_output_file(self, slug: str, filename: str, dest_dir: Path) -> Path:
        if "/" not in slug:
            slug = f"{self.whoami()}/{slug}"
        user, kslug = slug.split("/", 1)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / Path(filename).name
        with (
            self._client() as c,
            c.stream(
                "GET",
                f"/api/v1/kernels/output/download/{user}/{kslug}/{filename}",
                follow_redirects=True,  # 302 -> signed kaggleusercontent URL
            ) as r,
        ):
            if r.status_code != 200:
                raise KaggleError(f"output download {r.status_code} for {filename}")
            with open(dest, "wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)
        return dest

    def run_kernel(
        self,
        slug: str,
        title: str,
        script: str,
        *,
        gpu: bool = True,
        internet: bool = True,
        download_files: bool = True,
        out_dir: Path | None = None,
        run_timeout_s: float = DEFAULT_RUN_TIMEOUT_S,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    ) -> KernelRunResult:
        """Push -> poll to a terminal status -> harvest log (+ files).

        Poll errors raise; a terminal `error` status raises with the tail of
        the kernel log so failures surface instead of hanging the pipeline.
        """
        pushed = self.push_kernel(
            slug,
            title,
            script,
            gpu=gpu,
            internet=internet,
            session_timeout_s=int(run_timeout_s) if run_timeout_s else None,
        )
        full_slug = pushed.get("ref", "").removeprefix("/code/") or slug
        if "/" not in full_slug:
            full_slug = f"{self.whoami()}/{slug}"
        deadline = time.monotonic() + run_timeout_s
        status = "queued"
        while time.monotonic() < deadline:
            time.sleep(poll_interval_s)
            status = self.kernel_status(full_slug)
            if status in TERMINAL_STATUSES:
                break
        if status not in TERMINAL_STATUSES:
            raise KaggleError(
                f"kernel {full_slug} timed out after {run_timeout_s}s (last: {status})"
            )
        out = self.kernel_output(full_slug)
        if status == "error":
            tail = "".join(str(rec.get("data", "")) for rec in out["log"])[-800:]
            raise KaggleError(f"kernel {full_slug} failed:\n{tail}")
        files: list[str] = []
        if download_files and out["files"] and out_dir is not None:
            for entry in out["files"]:
                fname = entry["name"] if isinstance(entry, dict) else str(entry)
                self.download_output_file(full_slug, fname, Path(out_dir))
                files.append(fname)
        return KernelRunResult(
            slug=full_slug,
            url=pushed.get("url", f"{self.base_url}/code/{full_slug}"),
            status=status,
            log=out["log"],
            files=files,
            out_dir=Path(out_dir) if out_dir is not None else None,
        )


def kaggle_client_from_env() -> KaggleClient:
    """Build from settings — raises KaggleNotConfigured without credentials."""
    s = get_settings()
    if not s.kaggle_key:
        raise KaggleNotConfigured(
            "Kaggle not configured: set KAGGLE_KEY=KGAT_... (https://www.kaggle.com/settings/account)"
        )
    return KaggleClient(username=s.kaggle_username, key=s.kaggle_key)
