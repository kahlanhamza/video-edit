# gpu_services — GPU-adjacent services (Milestone 4)

The CPU-side render engine never calls heavy runtimes directly; it consumes their
*outputs* (receipts, tracks, stems) persisted by the perception layer, so local CI
never needs a GPU. Torch only ever lives inside these containers.

## Shipped

- **`sam2/` — SAM 2 local weight service** (`docker compose --profile gpu up -d --build sam2`):
  promptable video segmentation for the Studio's click-to-track loop. See
  [sam2/README.md](sam2/README.md). Client: `vision_fx.selection.Sam2ServiceBackend`
  (`SAM2_SERVICE_URL`).

## Planned (PLAN §4.4–§4.6)

WhisperX ASR fallback, Demucs (`htdemucs`) music stripping, DeepFilterNet speech
denoise, Ultralytics YOLO + ByteTrack face/plate blur, Qwen2.5-VL / Gemini frame
inspection — same shape as sam2: own container, own weights volume, HTTP port, hermetic tests.
