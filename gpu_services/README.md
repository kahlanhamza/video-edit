# gpu_services — Modal serverless GPU fleet (Milestone 4)

Not started. Planned endpoints (PLAN §4.4–§4.6): WhisperX ASR fallback, Demucs
(`htdemucs`) music stripping, DeepFilterNet speech denoise, Ultralytics YOLO + ByteTrack
face/plate blur, Qwen2.5-VL / Gemini frame inspection.

The CPU-side render engine never calls these directly; it consumes their *outputs*
(receipts, tracks, stems) persisted by the perception layer, so local CI never needs a GPU.
