FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# deep-filter (DeepFilterNet3) — the Bus-1 neural speech extractor for the
# 2-Bus Speech Isolation law. The release binary is static-pie (musl) with the
# model EMBEDDED, so a plain copy is the whole install: no runtime downloads.
ARG DEEPFILTER_VERSION=0.5.6
RUN curl -fsSL \
      "https://github.com/Rikorose/DeepFilterNet/releases/download/v${DEEPFILTER_VERSION}/deep-filter-${DEEPFILTER_VERSION}-x86_64-unknown-linux-musl" \
      -o /usr/local/bin/deep-filter \
    && chmod +x /usr/local/bin/deep-filter \
    && deep-filter --version

RUN pip install --no-cache-dir uv

# PaddleOCR — the LOCAL $0 overlay-OCR lane (perception.timestamp_ocr backend
# "paddle", issue #12). Optional by the cost law: the pipeline falls back to
# the keyed Gemini lane when absent. Uncomment to bake the CPU engine in:
# RUN uv pip install --no-cache-dir paddleocr==2.7.3

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages ./packages
COPY apps ./apps
RUN uv sync --frozen --no-dev

CMD ["/app/.venv/bin/celery", "-A", "director_os.worker.celery_app.celery_app", "worker", "-l", "info"]
