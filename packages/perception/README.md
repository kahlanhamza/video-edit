# packages/perception — multimodal perception layer (M2/M4)

Not started in M1. Owns the unified **Multimodal Timeline Index** (PLAN §3.2, §4.4):

- `perception.detect_shots` — PySceneDetect (`ContentDetector` + TransNetV2)
- `perception.index_visual_captions` — Stage-1 cheap VLM captioning + Laplacian
  blur/darkness/obstruction scoring
- `perception.inspect_frames` — "Frames-on-Demand" high-res frame pulls
- `perception.detect_audio_events` — YAMNet / PANNs (sirens, shouting, cuffs, gunshots)
- `perception.transcribe_diarize` — Gladia / WhisperX word-level timestamps + speakers
- `perception.sync_multicam` — `syncstart` audio cross-correlation (1 ms)
- `perception.ocr_burnins` — PaddleOCR on `AXON BODY 3` overlays and street signs

Everything writes into `perception_index` (see `director_os.core.db`).
