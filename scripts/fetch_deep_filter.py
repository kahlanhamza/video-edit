#!/usr/bin/env python3
"""Fetch the deep-filter release binary for THIS machine (the 2-bus enhancer).

The worker's 2-bus speech isolation needs the `deep-filter` Rust CLI
(DeepFilterNet3, MIT, model embedded in the release binary). Docker images
install it at build time (apps/worker/Dockerfile) — this script covers every
OTHER way of running the stack (host python, venv, opencode sandboxes on
Windows/macOS), where the pipeline otherwise honestly skips audio_prep with
"no neural enhancer installed".

Usage:
    python scripts/fetch_deep_filter.py            # -> ./bin/deep-filter[.exe]
    python scripts/fetch_deep_filter.py --dir /opt/tools --version 0.5.6

After it prints success, point the env at the binary and re-run the job:

    .env:     DEEPFILTER_BIN=/absolute/path/to/bin/deep-filter(.exe)
    then:     Run again — finished stages are never redone; only the audio
              is upgraded (isolate_2bus -> real neural enhancement).

Pure stdlib (urllib) — no pip installs, no admin rights, one static binary.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path

DEFAULT_VERSION = "0.5.6"

# (system, machine) -> release asset suffix
_ASSET_SUFFIXES = {
    ("Linux", "x86_64"): "x86_64-unknown-linux-musl",
    ("Linux", "aarch64"): "aarch64-unknown-linux-musl",
    ("Windows", "AMD64"): "x86_64-pc-windows-msvc.exe",
    ("Darwin", "x86_64"): "x86_64-apple-darwin",
    ("Darwin", "arm64"): "aarch64-apple-darwin",
}


def _asset_name(version: str) -> str:
    key = (platform.system(), platform.machine())
    suffix = _ASSET_SUFFIXES.get(key)
    if suffix is None:
        raise SystemExit(
            f"no prebuilt deep-filter release for {key[0]}/{key[1]} — build from "
            "source per https://github.com/Rikorose/DeepFilterNet (cargo install "
            "deep-filter --locked) and set DEEPFILTER_BIN to the built binary"
        )
    return f"deep-filter-{version}-{suffix}"


def _download(url: str, dest: Path, tok: int = 0) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "DirectorOS-fetch-deep-filter/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            if total and (got >> 20) >= tok:
                tok = (got >> 20) + 8
                print(f"    {got >> 20} / {total >> 20} MiB", flush=True)
    if total and got != total:
        dest.unlink(missing_ok=True)
        raise SystemExit(f"download truncated ({got} of {total} bytes) — retry")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="bin", help="target directory (default ./bin)")
    ap.add_argument(
        "--version", default=DEFAULT_VERSION, help=f"release tag (default {DEFAULT_VERSION})"
    )
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    version = args.version.removeprefix("v")
    asset = _asset_name(version)
    target_dir = Path(args.dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    exe_name = "deep-filter.exe" if platform.system() == "Windows" else "deep-filter"
    dest = target_dir / exe_name

    url = f"https://github.com/Rikorose/DeepFilterNet/releases/download/v{version}/{asset}"

    if dest.exists() and not args.force:
        print(f"[ok] {dest} already exists (use --force to re-fetch)")
    else:
        print(f"[fetch] {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        _download(url, tmp)
        tmp.replace(dest)
        if platform.system() != "Windows":
            dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"[sha256] {_sha256(dest)}")

    r = subprocess.run([str(dest), "--version"], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[warn] {dest} downloaded but --version exited {r.returncode}: {r.stderr[:200]}")
        return 1

    env_line = f"DEEPFILTER_BIN={dest}"
    if platform.system() == "Windows":
        env_line = f"DEEPFILTER_BIN={dest}".replace("\\", "/")
    print()
    print("deep-filter is ready. Add this line to your .env (Settings also works):")
    print(f"    {env_line}")
    print("Then press Run again in the Studio — finished stages are never redone;")
    print("only audio_prep + render re-execute with real neural speech isolation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
