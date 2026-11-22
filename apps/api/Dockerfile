FROM python:3.12-slim

# webrtcvad==2.0.10 has no cp312 wheel -> needs a C toolchain to build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages ./packages
COPY apps ./apps
RUN uv sync --frozen --no-dev

EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "director_os.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
