.PHONY: setup lint type test up down api worker render-fixture clean

setup:
        uv sync

lint:
        uv run ruff check .
        uv run ruff format --check .

type:
        uv run mypy packages apps

test:
        uv run pytest

up:
        docker compose up -d

down:
        docker compose down

api:
        uv run uvicorn director_os.api.main:app --reload --port 8000

worker:
        uv run celery -A director_os.worker.celery_app.celery_app worker -l info

render-fixture:
        uv run director render tests/fixtures/fixture_30s.mp4 --max-seconds 12 --preset 360p

demo-1080p:
        uv run python demo/demo_1080p.py

clean:
        rm -rf output .pytest_cache .mypy_cache .ruff_cache
