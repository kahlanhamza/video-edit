.PHONY: setup lint type test up down logs api worker render-fixture demo-1080p clean

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
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

# load the repo .env when present (shell env still wins), so a bare
# `make api` really is a working Studio - Gladia/LLM keys wire themselves
define uv_run_env
@if [ -f .env ]; then \
	uv run --env-file .env $(1); \
else \
	uv run $(1); \
fi
endef

api:
	$(call uv_run_env,uvicorn director_os.api.main:app --reload --port 8000)

worker:
	$(call uv_run_env,celery -A director_os.worker.celery_app.celery_app worker -l info)

render-fixture:
	uv run director render tests/fixtures/fixture_30s.mp4 --max-seconds 12 --preset 360p

demo-1080p:
	uv run python demo/demo_1080p.py

clean:
	rm -rf output .pytest_cache .mypy_cache .ruff_cache
