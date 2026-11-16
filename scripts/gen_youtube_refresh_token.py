"""Generate YOUTUBE_REFRESH_TOKEN for the YouTube Data API v3 publisher.

One-time interactive step — MUST run on a machine with a browser (laptop/desktop).
Google removed the out-of-band (OOB) flow in 2022, so this cannot be done on a
headless server: InstalledAppFlow spins up a tiny local HTTP server and your
browser completes the consent screen against it.

The OAuth client is read from the repo's .env (kept out of git on purpose):
    YOUTUBE_CLIENT_ID=...
    YOUTUBE_CLIENT_SECRET=...

Usage (from the repo root):
    pip install google-auth-oauthlib
    python scripts/gen_youtube_refresh_token.py

It prints the refresh token and offers to write it straight into .env.

Scopes match apps/api/src/director_os/api/youtube.py:
  - youtube.upload    (draft uploads)
  - youtube.force-ssl (playlist/listing management)

Notes:
  - access_type="offline" + prompt="consent" is what makes Google return a
    refresh token; without prompt="consent" an already-approved client may
    return none.
  - If the OAuth consent screen is in "Testing" status, add your Google
    account as a test user or the consent screen will refuse.
  - GitHub push protection treats OAuth client IDs as secrets, so they only
    ever live in .env — never hardcode them back into this script.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"


def _load_env() -> dict[str, str]:
    """Parse .env from the repo root (no python-dotenv dependency)."""
    if not ENV_PATH.exists():
        return {}
    out: dict[str, str] = {}
    for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def main() -> None:
    env = {**_load_env(), **{k: v for k, v in os.environ.items() if k.startswith("YOUTUBE_")}}
    client_id = env.get("YOUTUBE_CLIENT_ID", "")
    client_secret = env.get("YOUTUBE_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        print("Missing YOUTUBE_CLIENT_ID / YOUTUBE_CLIENT_SECRET.")
        print("Add them to .env (or export them) first — see .env.example.")
        sys.exit(2)

    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }

    flow = InstalledAppFlow.from_client_config(client_config, scopes=SCOPES)
    creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")

    print()
    print("=" * 60)
    print("YOUTUBE_REFRESH_TOKEN=" + str(creds.refresh_token))
    print("=" * 60)

    if not ENV_PATH.exists():
        print(f"(no {ENV_PATH} found — copy the token above manually)")
        return
    answer = input(f"\nWrite it into {ENV_PATH}? [y/N] ").strip().lower()
    if answer != "y":
        print("skipped — copy the token above manually")
        return

    lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    replaced = False
    for line in lines:
        if line.startswith("YOUTUBE_REFRESH_TOKEN="):
            out.append("YOUTUBE_REFRESH_TOKEN=" + str(creds.refresh_token))
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append("YOUTUBE_REFRESH_TOKEN=" + str(creds.refresh_token))
    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"done — {ENV_PATH} updated")
    sys.exit(0)


if __name__ == "__main__":
    main()
