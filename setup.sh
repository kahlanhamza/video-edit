#!/usr/bin/env sh
# DirectorOS one-command setup — macOS / Linux / WSL / Git Bash.
#
#   ./setup.sh            interactive (asks for the 2 keys, Enter skips)
#   ./setup.sh --yes      non-interactive (just boots the stack)
#
# After it finishes: Director Studio -> http://localhost:3000
# Add or change keys any time: edit .env, then: docker compose up -d

set -eu

cd "$(dirname "$0")"

ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    --yes|-y) ASSUME_YES=1 ;;
    *) echo "unknown option: $arg (supported: --yes)"; exit 2 ;;
  esac
done

line() { printf '%s\n' "-----------------------------------------------------------"; }

line
echo " DirectorOS — ship it"
line

# --- 1) prerequisites -------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: Docker is not installed."
  echo "  Install Docker Desktop (includes compose): https://docs.docker.com/get-docker/"
  echo "  Linux without Desktop: https://docs.docker.com/engine/install/"
  exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "ERROR: the 'docker compose' plugin is missing (needs Compose v2.24+)."
  echo "  Docker Desktop ships it; on Linux: docker-compose-plugin package."
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker is installed but not running (or you lack permission)."
  echo "  Start Docker Desktop / the docker daemon, then re-run ./setup.sh"
  exit 1
fi
echo "[1/4] Docker OK"

# --- 2) .env ----------------------------------------------------------------
if [ ! -f .env ]; then
  cp .env.example .env
  echo "[2/4] Created .env from .env.example"
else
  echo "[2/4] .env already exists — keeping your values"
fi

set_key() { # set_key FILE KEY VALUE   (portable sed -i across mac/linux)
  f="$1"; k="$2"; v="$3"
  if [ "$(uname)" = "Darwin" ]; then
    sed -i '' "s|^${k}=.*|${k}=${v}|" "$f"
  else
    sed -i "s|^${k}=.*|${k}=${v}|" "$f"
  fi
}

ask_key() { # ask_key KEY HINT  -> writes into .env unless skipped/kept
  k="$1"; hint="$2"
  cur=$(grep -E "^${k}=" .env | tail -n1 | cut -d= -f2-)
  if [ "$ASSUME_YES" = "1" ]; then
    [ -n "$cur" ] && echo "      $k already set — kept" || echo "      $k skipped (add later in .env)"
    return 0
  fi
  printf '\n  %s\n' "$hint"
  if [ -n "$cur" ]; then
    printf '  %s is already set. New value (Enter to keep): ' "$k"
  else
    printf '  Paste %s (Enter to skip): ' "$k"
  fi
  IFS= read -r answer || answer=""
  if [ -n "$answer" ]; then
    set_key .env "$k" "$answer"
    echo "      $k saved to .env"
  else
    echo "      $k kept as-is"
  fi
}

echo "--- API keys (both optional; Enter skips) ---"
ask_key GLADIA_API_KEY "Gladia = transcription/captions. Free key: https://app.gladia.io"
ask_key LLM_API_KEY    "LLM = the AI editor's brain. Meta Model API key: https://fai.meta.com/developer (or a Google AI Studio key: https://aistudio.google.com/apikey)"

# SETTINGS_SECRET_KEY: encrypts the settings vault at rest (CR-008). Generate
# one automatically unless the user already set it — an empty key means API
# keys and the YouTube refresh token sit PLAINTEXT in the database.
cur_secret=$(grep -E "^SETTINGS_SECRET_KEY=" .env | tail -n1 | cut -d= -f2-)
if [ -z "$cur_secret" ]; then
  gen=""
  if command -v python3 >/dev/null 2>&1; then
    gen=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))" 2>/dev/null)
  fi
  [ -z "$gen" ] && command -v openssl >/dev/null 2>&1 && gen=$(openssl rand -base64 32 | tr -d '\n=/+' | head -c 43)
  if [ -n "$gen" ]; then
    set_key .env SETTINGS_SECRET_KEY "$gen"
    echo "      SETTINGS_SECRET_KEY generated — saved provider keys are encrypted at rest"
  else
    echo "      WARNING: could not generate SETTINGS_SECRET_KEY — the vault stays plaintext"
  fi
else
  echo "      SETTINGS_SECRET_KEY already set — kept"
fi
echo "[2/4] Keys configured"

# --- 3) build + boot --------------------------------------------------------
echo "[3/4] Building and starting the stack (first build takes a few minutes)..."
docker compose up -d --build

# --- 4) health wait ---------------------------------------------------------
echo "[4/4] Waiting for the API and Studio to come up..."
API_UP=0; WEB_UP=0
i=0
while [ $i -lt 60 ]; do
  if [ "$API_UP" = "0" ] && curl -sf http://localhost:8000/health >/dev/null 2>&1; then API_UP=1; fi
  if [ "$WEB_UP" = "0" ] && curl -sf http://localhost:3000 >/dev/null 2>&1; then WEB_UP=1; fi
  [ "$API_UP" = "1" ] && [ "$WEB_UP" = "1" ] && break
  i=$((i+1)); sleep 3
done

line
if [ "$API_UP" = "1" ] && [ "$WEB_UP" = "1" ]; then
  echo " SHIPPED. Everything is up:"
else
  echo " Stack is starting (or something is slow). Check:"
  echo "   docker compose ps"
  echo "   docker compose logs -f api"
fi
echo
echo "  Director Studio (the web app) : http://localhost:3000"
echo "  API gateway                   : http://localhost:8000  (docs: /docs)"
echo "  MinIO console                 : http://localhost:9001  (directoros / directoros-secret)"
echo
echo "  Use it: open http://localhost:3000 -> New case -> paste a video URL"
echo "  (+ optional article/news/PDF links) -> Create. The pipeline runs"
echo "  itself and parks the master at the Human Review Gate."
echo
echo "  Keys later: edit .env, then: docker compose up -d"
echo "  Logs:       docker compose logs -f"
echo "  Stop:       docker compose down            (data is kept)"
line
