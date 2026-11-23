# SAM 2 local weight service — the ONLY container in this repo allowed to
# carry torch. CPU-first (sam2.1 Hiera-Tiny tracks fine on CPU for the spans
# the Studio prompts); set deploy.resources or an NVIDIA runtime for GPU.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    SAM2_WEIGHTS_DIR=/weights \
    PIP_NO_CACHE_DIR=1

# ffmpeg: the service never encodes, but ultralytics' video predictor probes
# frame counts via ffprobe; keep the stack honest end to end.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY service.py .

# weights live on a named volume: download once, restart forever
VOLUME ["/weights"]

EXPOSE 8300
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -sf http://localhost:8300/health || exit 1

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8300"]
