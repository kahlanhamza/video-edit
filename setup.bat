@echo off
rem DirectorOS one-command setup - Windows (cmd / PowerShell-compatible).
rem Equivalent of setup.sh: create .env, build and boot the whole stack.
rem Add your keys any time: edit .env, then run: docker compose up -d

cd /d "%~dp0"

where docker >nul 2>nul
if errorlevel 1 (
  echo ERROR: Docker is not installed or not on PATH.
  echo   Install Docker Desktop: https://docs.docker.com/get-docker/
  exit /b 1
)

docker info >nul 2>nul
if errorlevel 1 (
  echo ERROR: Docker is installed but not running. Start Docker Desktop first.
  exit /b 1
)

if not exist .env (
  copy .env.example .env >nul
  echo Created .env from .env.example
) else (
  echo .env already exists - keeping your values
)

echo.
echo Optional API keys (edit .env after this script if you skip now):
echo   GLADIA_API_KEY  - transcription/captions  ^<https://app.gladia.io^>
echo   LLM_API_KEY     - the AI editor brain    ^<https://fai.meta.com/developer^>
echo.

echo Building and starting the stack ^(first build takes a few minutes^)...
docker compose up -d --build

echo.
echo When everything is healthy:
echo   Director Studio : http://localhost:3000
echo   API gateway     : http://localhost:8000  ^(docs at /docs^)
echo.
echo Stop: docker compose down        Logs: docker compose logs -f
