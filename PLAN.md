# Project Blueprint: Autonomous Multimodal Video Editing & Documentary Platform ("DirectorOS")

> **Status:** Architecture RFC / Master Implementation Plan (v1.0)  
> **Target Domain:** Long-form Documentary, Commentary & Police Bodycam Post-Production (12–20 min 16:9, scalable to Shorts/Multi-Niche)  
> **Core Principle:** Durable Cloud State + Multimodal Perception + Native Agentic Tool Execution + `OpenTimelineIO` Backbone + Human-in-the-Loop Studio

---

## 1. Executive Summary & Why We Moved Beyond Notebooks

Our initial Google Colab prototype validated the core editorial formula (raw public records → cloud ASR + diarization → AI documentary script → pause-trimmed neural TTS → speech-aligned cuts → 2-row speaker-colored `.ass` subtitles → corner blur + glass graphic cards). However, ephemeral notebooks fail in production because:

1. **Zero Durability:** Container recycling wipes raw media, intermediate `.mp4` chunks, and `.json` word timings.
2. **Blind Text-Only Editing:** A text-only LLM misses silent visual gold (e.g., an officer discovering contraband silently) and keeps dark/obstructed shots.
3. **Full Re-Renders on Small Fixes:** Fixing one word or one cut forces a full 12-minute re-encode.
4. **No Visual Review UI:** Checking watermark coordinates or fixing 400+ transcript lines in raw text files does not scale.
5. **No Multi-Cam / Copyright Safeguards:** Real FOIA packets contain 3–6 unsynced cameras and background copyrighted music (car radios, store speakers).

**DirectorOS** solves this by treating video production as a **durable, event-driven state machine** where multimodal AI agents operate directly on an industry-standard **`OpenTimelineIO (.otio)`** timeline through a **strongly-typed Native Tool Execution Engine**.

---

## 2. High-Level System Architecture

```text
+-----------------------------------------------------------------------------------+
|                        1. DIRECTOR STUDIO (Next.js 16 Web UI)                     |
|  [URL/FOIA Ingest]  [Script & Cut Editor]  [2-Row Subtitle Grid]  [Blur Canvas]   |
+-----------------------------------------+-----------------------------------------+
                                          | REST / SSE / WebSockets
+-----------------------------------------v-----------------------------------------+
|                       2. API GATEWAY & ORCHESTRATION LAYER                        |
|        FastAPI (Python 3.12+)  <-->  Temporal / Celery+Redis Workflow Engine      |
|        PostgreSQL 16 (+ pgvector)    |    Cloudflare R2 / MinIO (S3 Storage)      |
+-----------------------------------------+-----------------------------------------+
                                          | Typed Tool Calls & Checkpoints
+-----------------------------------------v-----------------------------------------+
|                  3. NATIVE AGENTIC TOOL RUNTIME ("MCP-Style Engine")              |
|  [Perceiver Agent] ---> [Director Agent] ---> [Assembler Agent] ---> [QA Critic]  |
|   Tool Registry (Pydantic v2 Schemas) | OTIO Snapshot/Undo | Guardrail Sandbox    |
+-----------------------------------------+-----------------------------------------+
                                          | Dispatches CPU & Serverless GPU Jobs
+-----------------------------------------v-----------------------------------------+
|                       4. SPECIALIZED MEDIA & AI WORKER FLEET                      |
|  CPU Workers (VPS/Docker): FFmpeg, libass, Pillow, Playwright, OTIO, Edge-TTS     |
|  GPU Workers (Modal Serverless): WhisperX, PyAnnote, Demucs, YOLO Blur, VLM/CLIP  |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
              [Cloud-to-YouTube Draft Upload (YouTube Data API v3) + .OTIO Export]
```

---

## 3. Native Agentic Tool Execution Engine ("MCP-Style Without MCP")

### 3.1 Architectural Philosophy
Instead of relying on an external desktop MCP client (e.g., Claude Desktop over stdio/SSE) to drive an editor, **DirectorOS embeds an internal, first-class Agentic Tool Runtime directly inside the backend**.

* **How it works:** The AI Director and Editor agents interact with the entire media pipeline via **native LLM Function/Tool Calling** (OpenAI / Anthropic / Gemini structured tool specs), routed through a central **Tool Registry & Dispatcher**.
* **Why this beats external MCP for production:**
  1. **Zero external client dependency:** Runs autonomously 24/7 in background cloud workflows.
  2. **Transactional State + Instant Undo:** Every timeline-mutating tool writes an immutable **`OTIO` snapshot** in PostgreSQL/R2 before executing. Any bad AI decision can be rolled back to `checkpoint_id` in 5ms.
  3. **Deterministic Guardrails:** The LLM proposes creative tool calls; deterministic Python validators enforce hard physical, audio, and monetization laws *before* committing (e.g., preventing overlapping sample rates, bounds errors, or unblurred bystander faces).
  4. **Dynamic Skill Injection (`SKILL.md`):** Niche playbooks (e.g., `bodycam_documentary.md`, `shoplifting_commentary.md`, `courtroom_breakdown.md`) are loaded as modular system instructions telling the agents *how* to pace, narrate, and style each format.
  5. **Optional External MCP Facade:** Because every capability is already a schema-validated Python tool, exposing a thin `FastMCP` adapter later takes <100 lines if a human ever wants to attach an external IDE/assistant.

### 3.2 The 4-Agent Hierarchy
1. **Perceiver Agent (Multimodal Analyst):** Never edits. Converts raw video/audio/PDFs into the unified **Multimodal Timeline Index** (speech + visual scene descriptions + audio events + shot quality + multi-cam offsets).
2. **Director Agent (Creative Storyteller):** Reasons over retention, story arc, police procedures, and contradictions. Selects clips, switches cameras, writes narrator bridges, and plans J/L-cuts, punch-in zooms, and freeze-callouts.
3. **Assembler Agent (Precision Technician):** Translates creative decisions into exact **OTIO** mutations, audio-ducked mixes, corrected 2-row `.ass` subtitles, blur boxes, and graphics cards.
4. **Critic / QA Agent (Automated Reviewer):** Audits rendered previews deterministically (black frames, clipping, card/face collisions) and visually via VLM frame inspection before unlocking the Human Review Gate.

### 3.3 Complete Internal Tool Catalog (Function-Calling Registry)

#### Namespace A: `ingest.*`
* `ingest.fetch_media(source_url, custom_headers, project_id)` — Cloud-side download with custom `User-Agent`/cookies (bypasses Wikimedia/portal scraper blocks).
* `ingest.parse_case_docs(pdf_urls_or_files)` — Extracts arrest reports, charges, dates, and suspect details from FOIA/COPA PDFs.
* `ingest.build_proxies(asset_id)` — Generates lightweight 480p web-playable proxy + thumbnail sprite strip ("scrubsheet") + `64k` ASR audio.

#### Namespace B: `perception.*`
* `perception.detect_shots(asset_id, detector="transnetv2|content")` — Finds hard cuts and gradual transitions.
* `perception.index_visual_captions(asset_id, sample_strategy="shot_mid_and_5s")` — Runs Stage-1 cheap VLM visual captioning + Laplacian blur/darkness/obstruction scoring.
* `perception.inspect_frames(asset_id, timestamps[], purpose)` — **"Frames-on-Demand"**: pulls high-res frames at exact timestamps for the AI to visually inspect hands, contraband, face positions, and watermarks before cutting or placing cards.
* `perception.detect_audio_events(asset_id)` — Tags sirens, shouting, handcuffs, impacts, gunshots, and silence.
* `perception.transcribe_diarize(asset_id, provider="gladia|whisperx|deepgram")` — Produces word-level millisecond timestamps + speaker IDs.
* `perception.sync_multicam(asset_ids[])` — Audio cross-correlation down to 1ms + clock-drift measurement across multiple bodycams/dashcams.
* `perception.ocr_burnins(asset_id, timestamps[])` — Reads native `AXON BODY 3` timestamps, squad numbers, or street signs.

#### Namespace C: `timeline.*` (Backed by `OpenTimelineIO` + Checkpoints)
* `timeline.create_checkpoint(project_id, label)` / `timeline.rollback(project_id, checkpoint_id)`
* `timeline.add_clip(track, asset_id, source_start_s, source_end_s, insert_at_s, metadata)`
* `timeline.trim_dead_air(track, clip_id, min_gap_s=1.8, pad_before_s=0.6, pad_after_s=0.8)` — Uses speech + audio-event + motion intervals (never blind highway volume!).
* `timeline.switch_camera(timeline_range, target_asset_id, reason)` — Switches between synced Officer 1, Officer 2, and Dashcam angles.
* `timeline.apply_jl_cut(clip_a_id, clip_b_id, mode="J|L", offset_s)` — Overlaps audio lead/lag across visual cuts.
* `timeline.split_clip(clip_id, at_s)` / `timeline.delete_clip(clip_id)` / `timeline.reorder_clips(order[])`
* `timeline.add_chapter_marker(at_s, title)` — Auto-populates YouTube chapters.

#### Namespace D: `audio.*`
* `audio.separate_stems(asset_id, range_s, model="htdemucs")` — Strips copyrighted background music (car radios/store PA) while preserving voices.
* `audio.denoise_speech(asset_id, range_s, engine="deepfilternet|noisereduce")` — Cleans wind/static noise on muffled mics.
* `audio.enhance_speech(asset_id, engine="auto|deepfilter|clearervoice|afftdn")` — the 2-Bus Speech Isolation pre-pass (`stage_audio_enhance`): 90 Hz rumble pre-clean → neural extractor → intelligibility EQ + compression → speech-gated −14 dB duck (Silero VAD + ASR words, 150 ms ramps) → tamed ambience bed (−24 dB, 150 Hz–4 kHz) → speech-gated gain → −1.5 dBTP limiter, 48 kHz stereo. Content-addressed per asset; writes the exact `enhancer_engine` into `agent_tool_logs`; `auto` NEVER silently falls back to afftdn.
* `audio.generate_tts(text, voice_id, provider="edge|kokoro|elevenlabs", rate, trim_pauses=True)` — Synthesizes narration + trims dead padding (`silenceremove`).
* `audio.mix_ducking(narration_track, bed_track, duck_db=-18)` — Applies smooth `sidechaincompress` or hard-mute (`volume=0`) under narrator segments.
* `audio.normalize_master(target_lufs=-16.0, sample_rate=48000)` — Enforces uniform `48 kHz stereo` across every segment.

#### Namespace E: `vision_fx.*` & `redact.*`
* `vision_fx.punch_in_zoom(clip_id, start_s, end_s, zoom_factor=1.25, target_bbox="auto_face|manual")` — Smooth keyframed zoom on speaker/evidence.
* `vision_fx.freeze_callout(clip_id, freeze_at_s, hold_s, dim_alpha=0.45, spotlight_bbox, narration_id)` — Freezes frame, dims background, highlights evidence while narrator explains.
* `redact.detect_sensitive_regions(asset_id, classes=["face", "license_plate"])` — Runs YOLO + ByteTrack tracking across frames.
* `redact.add_blur_region(scope="global|clip|range", bbox_xywh, mode="static|tracked", sigma=24)` — Masks corner channel watermarks, minors, bystanders, IDs, or plates.

#### Namespace F: `subtitle.*` & `graphics.*`
* `subtitle.correct_and_tag(transcript_id, corrections[], speaker_map)` — LLM + rules fix police codes (`10-8`, `10-15`, `Signal Zero`, `Standard Field Sobriety`) and label `Officer` vs. `Suspect` vs. `Narrator`.
* `subtitle.align_words(corrected_text, raw_words[])` — `difflib.SequenceMatcher` alignment preserving millisecond precision even after rewrites.
* `subtitle.build_ass(style_preset="doc_2row_karaoke", profanity_mode="mask")` — Generates 3–5 word beat `.ass` file with 2-row simultaneous speaker lanes.
* `graphics.render_card(template="location_intro|case_outcome|map_route|evidence_quote", fields_json, start_s, duration_s)` — Renders anti-aliased RGBA PNG/WebM overlays and places them in a collision-free corner.

#### Namespace G: `render.*`, `qa.*` & `publish.*`
* `render.preview_frame(project_id, at_s)` / `render.preview_clip(project_id, start_s, end_s)` — 1–5s instant visual check before full render.
* `render.build_final(project_id, resolution="1080p", use_segment_cache=True)` — Hash-caches unchanged segments, renders dirty segments, concatenates at uniform codec/sample rate.
* `render.export_nle(project_id, format="otio|fcpxml|edl")` — Exports editable timeline for DaVinci Resolve / Premiere / Kdenlive.
* `qa.audit_technical(video_path)` — Checks black frames, freeze frames, loudness/clipping, and audio sync.
* `qa.audit_layout_and_policy(project_id)` — Verifies cards/subtitles don't overlap faces or native `AXON` timestamps, checks unmasked watermarks, and flags graphic/profanity monetization risks.
* `publish.upload_youtube_draft(video_path, srt_path, thumbnail_path, metadata_json, privacy="private")` — Cloud-to-YouTube upload with chapters and captions.

---

## 4. Complete Tooling & Technology Matrix (Every Component + Alternatives)

### 4.1 Frontend ("Director Studio" Web App)
| Component | Primary Choice | Alternative Options | Purpose |
| :--- | :--- | :--- | :--- |
| **Framework & Language** | `Next.js 16` (App Router) + `React 19` + `TypeScript` | `React + Vite + TanStack Router`, `SvelteKit` | Full web dashboard & SSR |
| **Styling & UI Kit** | `Tailwind CSS v4` + `shadcn/ui` + `Lucide React` | `Radix UI + CSS Modules`, `Mantine` | Clean dark-mode NLE aesthetic |
| **Client State & Queries** | `TanStack Query v5` + `Zustand` | `Redux Toolkit`, `Jotai` | Async job polling + timeline undo state |
| **Video Proxy Player** | `Media Chrome` / HTML5 `<video>` + custom scrub bar | `Video.js`, `Plyr` | Frame-accurate proxy playback |
| **Audio Waveform** | `WaveSurfer.js` | `Peaks.js` | Visualizing speech vs. dead air & narration ducks |
| **Interactive Blur/Card Canvas** | `react-konva` (`Konva.js`) | `Fabric.js`, raw HTML5 `<canvas>` | Mouse-drag watermark blur boxes & card placement |
| **Subtitle Grid Editor** | `TanStack Table` | `AG Grid` | Fast inline editing of 400+ color-coded caption lines |
| **Live Job Progress** | `Server-Sent Events (SSE)` | `WebSockets` (`Socket.IO`), `Ably/Pusher` | Real-time `[14/26]` render & agent logs |

### 4.2 Backend API, Agent Harness & Orchestration
| Component | Primary Choice | Alternative Options | Purpose |
| :--- | :--- | :--- | :--- |
| **API Server** | `Python 3.12+` + `FastAPI` + `Uvicorn` | `Litestar`, Node `Fastify/Hono` (API only) | High-speed async REST + SSE |
| **Package & Env Manager** | `uv` (Astral) | `Poetry`, `pip-tools` | 10–100x faster deterministic Python builds |
| **Agent Harness & Tool Calling** | `Pydantic AI` (`pydantic-ai` + `pydantic-graph`) | `LangGraph` (built-in Postgres checkpointer & `interrupt()`), `OpenAI Agents SDK`, `Claude Agent SDK` | Strongly-typed `@agent.tool` execution & validation |
| **Workflow Orchestrator** | `Temporal` (`temporalio` Python SDK) — *Big-Tech Gold Standard* | `Celery 5 + Redis` (lighter single-VPS Python classic), `Trigger.dev` (TS-first), `Prefect` | Durable multi-day HITL pauses, retries, crash recovery |
| **Validation & Resilience Helpers** | `Pydantic v2`, `httpx`, `tenacity`, `structlog` | `attrs`, `aiohttp`, `backoff` | Schema contracts, async HTTP, retries, JSON logs |

### 4.3 Databases, Search & Object Storage
| Component | Primary Choice | Alternative Options | Purpose |
| :--- | :--- | :--- | :--- |
| **Relational + Vector DB** | `PostgreSQL 16+` with `pgvector` extension | `Supabase` / `Neon` (managed Postgres), `Qdrant` / `Milvus` (dedicated vector DB at scale) | Projects, OTIO snapshots, transcripts, scene embeddings |
| **ORM & Migrations** | `SQLAlchemy 2.0` (async `asyncpg`) + `Alembic` | `SQLModel`, `Prisma` (if TS API) | Versioned schema management |
| **Cache & Pub/Sub** | `Redis 7` | `Valkey`, `Dragonfly` | Progress streams, rate limits, Celery broker |
| **Object Storage** | `Cloudflare R2` ($0 egress, 10 GB/mo free; card required) | `MinIO` (100% self-hosted S3, no card), `Backblaze B2`, `AWS S3` | Raw video, proxies, cached chunks, final MP4s |
| **Storage SDK** | `boto3` / `aioboto3` | `minio` Python client | Streaming S3 uploads/downloads |

### 4.4 Ingest, Multi-Cam Sync & Multimodal Perception
| Component | Primary Choice | Alternative Options | Purpose |
| :--- | :--- | :--- | :--- |
| **Media Fetching** | `yt-dlp` + `httpx` (custom `User-Agent`) | `curl_cffi`, `Playwright` (JS-gated portals) | Pulling public portals, YouTube, Wikimedia, COPA |
| **PDF Case Report Parsing** | `PyMuPDF (fitz)` + `pdfplumber` | `Unstructured`, `Docling` | Reading official arrest/incident reports |
| **Multi-Cam Audio Sync** | `syncstart` (`scipy.fft` cross-correlation + `z-score`) | Custom `librosa` + `scipy.signal.correlate` | 1ms alignment across multiple BWC/dashcams |
| **Shot Boundary Detection** | `PySceneDetect` (`ContentDetector` + `TransnetV2Detector`) | Standalone `TransNetV2` (PyTorch) | Detecting cuts & gradual transitions |
| **Fast Frame Extraction** | `decord` / `PyAV` (`av`) | `OpenCV (cv2)` | Random-access keyframe extraction without decoding full video |
| **Visual Embeddings & Search** | `SigLIP` / `OpenCLIP` (`open_clip_torch`) | `InternVideo2` | Semantic scene clustering & search in `pgvector` |
| **Multimodal VLM Analysis** | `Gemini 2.5 / 3 Flash` (`google-genai` SDK) | `TwelveLabs` (`Marengo` + `Pegasus` API), self-hosted `Qwen2.5-VL` / `Vidi-7B` | "Caption-once" scene index + "Frames-on-Demand" check |
| **On-Screen OCR** | `PaddleOCR` | `EasyOCR`, `Tesseract` (`pytesseract`) | Reading `AXON BODY 3` timestamps, car numbers, street signs |
| **Non-Speech Audio Events** | `YAMNet` / `PANNs` (`panns-inference`) | `AST` (Audio Spectrogram Transformer), `CLAP` | Detecting sirens, shouting, cuffs, gunshots, impacts |
| **Frame Quality Scoring** | `OpenCV` (Laplacian variance + HSV luma histograms) | `BRISQUE` | Rejecting dark/pocket/blurry bodycam angles |
| **GPU/Compute Execution** | `ComputeBackend` port (`vision_fx/compute`): `LocalOnnxCpuBackend` DEFAULT ($0: yolov8n-face ONNX stride-3 + interp, YuNet fallback) | `KaggleBatchGpuBackend` ($0 free T4 via `/api/v1/kernels/push`, `KGAT_*` bearer — heavy WhisperX/Demucs/YOLO); `ModalGpuBackend` (dormant behind `MODAL_TOKEN_ID`) | One `COMPUTE_BACKEND` env switch; tracked faces return persistent `person_N` IDs in RAW pixel space |

### 4.5 Speech Recognition (ASR), Audio Cleaning & Voiceover (TTS)
| Component | Primary Choice | Alternative Options | Purpose |
| :--- | :--- | :--- | :--- |
| **Cloud ASR + Diarization** | `Gladia API` (`solaria-3` + enhanced diarization) | `Deepgram` (`nova-3`), `AssemblyAI` (`universal-2`) | Fast noisy-audio word timestamps & speakers |
| **Self-Hosted GPU ASR** | `WhisperX` (`faster-whisper` + `wav2vec2` + `pyannote.audio 3.x` + `Silero VAD`) | `faster-whisper` standalone, `Insanely Fast Whisper` | Zero per-minute cost on Modal GPU |
| **Active Speaker Detection** | `Columbia ASD` / `TalkNet` + `MediaPipe` | Audio-diarization-only fallback | Confirming who is speaking when faces are visible |
| **Copyright Music Removal** | `Demucs` (`htdemucs`) | `audio-separator` (MDX-Net / RoFormer) | Stripping copyrighted car/store music while keeping dialogue |
| **Speech Noise Reduction** | `deep-filter` Rust CLI (`DeepFilterNet3`, model embedded in the release binary, `--atten-lim-db 28`) via the `SpeechEnhancer` port; `ClearerVoice MossFormer2_SE_48K` (Apache-2.0, native 48 kHz) when the `clearvoice` package is installed | `afftdn` (EXPLICIT weak fallback only — fails the production QA gate); `noisereduce`, `RNNoise` | Broadcast 2-Bus Speech Isolation: highway traffic, wind gusts, cruiser engine noise out of the voice bus |
| **Voiceover TTS (Pluggable)** | `TtsBackend` port (`audio_engine/tts.py`): `EdgeTtsBackend` DEFAULT ($0, no key) | `ElevenLabsTtsBackend` (pure httpx `/v1/text-to-speech/{voice_id}/with-timestamps` → word-level stamps; NOTE: free tier blocks library voices via API — set `ELEVENLABS_VOICE_ID` to an owned voice). **Local/Free GPU:** `Kokoro-82M` (`kokoro-onnx`), `Chatterbox`, `Qwen3-TTS` | Documentary narration; `TTS_PROVIDER` env flips providers, zero SDK |
| **Audio Mastering & Utilities** | `FFmpeg` (`silenceremove`, `loudnorm`, `sidechaincompress`, `acrossfade`) + `soundfile`, `numpy`, `scipy` | `pydub`, `sox` | Tight pacing, ducking & strict `48 kHz` stereo lock |

### 4.6 Computer Vision Redaction, Timeline, Subtitles & Rendering
| Component | Primary Choice | Alternative Options | Purpose |
| :--- | :--- | :--- | :--- |
| **Face & Plate Detection** | `Ultralytics YOLO` (`YOLOv8-face` + license-plate weights) | `MediaPipe Face Detection`, `RetinaFace`, `InsightFace` | Automated privacy redaction for FOIA footage |
| **Object Tracking & Smoothing** | `ByteTrack` / `BoT-SORT` (built into Ultralytics) + Kalman smoothing | `OpenCV CSRT / KCF`, `SAM 2` (Segment Anything 2) | Smooth non-flickering blur boxes across motion |
| **Canonical Timeline Engine** | `OpenTimelineIO` (`opentimelineio`, `opentime`, `otio-fcpx-xml-adapter`) | Custom JSON EDL (legacy) | Universal NLE interchange (Resolve, Premiere, Kdenlive, FCP) |
| **Subtitle Formatting & Alignment** | `pysubs2` + Python `difflib.SequenceMatcher` | `srt` / `webvtt-py` (plain only) | Millisecond-aligned 2-row color-coded `.ass` captions |
| **Subtitle Renderer** | `FFmpeg` + `libass` (`ass=`) + `fonts-roboto` | `Montserrat` / `Inter` / `Poppins` fonts | High-contrast active-word highlight rendering |
| **Graphics & Overlay Engine** | `Pillow (PIL)` (static glass cards) + `Playwright` (HTML/CSS/SVG maps, charts & docs → PNG/WebM) | `HyperFrames` (Apache-2.0 HTML-to-video), `Motion Canvas / Revideo` *(Avoid Remotion if scaling >3 seats due to paid Company License)* | Documentary cards, animated street maps, arrest highlights |
| **Core Video Engine** | `FFmpeg` / `FFprobe` (`libx264` CPU / `h264_nvenc` GPU) | `GStreamer` | Segment-cached rendering & `-c:v copy` fast muxing |

### 4.7 QA Guardrails, Publishing & Infrastructure
| Component | Primary Choice | Alternative Options | Purpose |
| :--- | :--- | :--- | :--- |
| **Technical QA** | `FFmpeg` (`blackdetect`, `freezedetect`, `silencedetect`, `ebur128`) | `PyAV` custom frame checks | Catching black frames, drift, clipping |
| **Policy & Profanity Filter** | `better-profanity` + custom YouTube advertiser regex | `Perspective API` | Censoring captions to avoid yellow `"$"` limited ads |
| **Music Leak Check (Optional)** | `pyacoustid` (`Chromaprint / fpcalc`) | `AudD` / `ACRCloud` API | Detecting leftover copyrighted music before upload |
| **YouTube Auto-Publishing** | `google-api-python-client` + `google-auth-oauthlib` (YouTube Data API v3) | Manual download from R2 | Direct cloud-to-YouTube Private Draft + Chapters + `.srt` |
| **Thumbnail Pipeline (Phase 4)** | `OpenCV` sharpest-frame picker + `rembg` + `Playwright/Pillow` | `Flux` / `Gemini Image` | High-CTR documentary thumbnails |
| **Serverless GPU Burst** | `Modal` (Starter: $30/mo free credits, T4/L4/A10G per-second billing) | `RunPod Serverless`, `Replicate`, `Beam` | Running YOLO, WhisperX, Demucs only when needed |
| **Core Hosting** | **Bootstrap:** `Oracle Cloud Always Free ARM` (2 OCPU / 12 GB RAM / 200 GB Storage) + `R2` + `Modal`<br>**Production:** `Hetzner Cloud` x86 NVMe VPS (`Docker Compose`) + `R2` + `Modal` | `DigitalOcean`, `AWS ECS / Fargate` | Cost-efficient 24/7 cloud execution |
| **Observability & Tracing** | `Sentry` + `Pydantic Logfire` (or `Langfuse`) + `OpenTelemetry` | `Prometheus + Grafana`, `Datadog` | Tracking LLM token spend, tool calls & render failures |
| **Testing, Linting & CI/CD** | `pytest` + `Playwright E2E` + `Ruff` + `mypy` + `ESLint` + `GitHub Actions` | `flake8`, `black`, `GitLab CI` | Preventing FFmpeg/OTIO regressions |

---

## 5. Canonical Data Models

### 5.1 PostgreSQL Schema (Core Tables)
* `projects`: `id (UUID)`, `title`, `niche_skill`, `status`, `active_otio_version_id`, `youtube_draft_id`, `created_at`
* `media_assets`: `id`, `project_id`, `source_url`, `role (raw_bwc|dashcam|911|pdf_report|music)`, `r2_raw_key`, `r2_proxy_key`, `r2_audio_key`, `duration_s`, `fps`, `width`, `height`, `sync_offset_ms`
* `perception_index`: `asset_id`, `transcript_json (JSONB)`, `shots_json (JSONB)`, `audio_events_json (JSONB)`, `redaction_tracks_json (JSONB)`, `scene_embeddings (vector(768))`
* `timeline_versions`: `id`, `project_id`, `version_num`, `otio_json (JSONB)`, `created_by (director_agent|human_edit)`, `checkpoint_note`, `created_at`
* `segment_cache`: `segment_hash (PK)`, `r2_chunk_key`, `duration_s`, `codec_profile`, `last_used_at` *(enables 2-second partial re-renders!)*
* `agent_tool_logs`: `id`, `project_id`, `agent_name`, `tool_name`, `args_json`, `result_summary`, `token_cost_usd`, `duration_ms`, `created_at`
* `jobs`: `id`, `project_id`, `temporal_workflow_id`, `current_stage`, `progress_pct`, `error_message`

### 5.2 `OpenTimelineIO` Custom Metadata Contract
Every `otio.schema.Clip` carries structured `metadata["director_os"]`:
* **`raw_bwc` clips:** `{"kind": "raw", "audio_mode": "loudnorm_48k", "trim_silence": true, "zoom_keyframes": [...], "blur_regions": [...], "angle_id": "bwc_1"}`
* **`narrator` clips:** `{"kind": "narration", "tts_provider": "edge", "voice_id": "en-US-BrianMultilingualNeural", "script_text": "...", "broll_asset_id": "...", "broll_range": [s, e], "broll_audio_volume": 0.0, "jl_cut": {"j_s": 0.4}}`
* **`overlay` track items:** `{"kind": "card", "template": "case_outcome", "fields": {...}, "fade_s": 0.5}`

---

## 6. Monorepo Folder Structure

```text
director-os/
├── PLAN.md
├── docker-compose.yml              # API, Temporal/Celery, Postgres+pgvector, Redis, MinIO, Next.js
├── pyproject.toml                  # Workspace config managed by uv
├── .github/workflows/ci.yml        # Ruff, mypy, pytest, Next.js build
│
├── apps/
│   ├── web/                        # Next.js 16 "Director Studio" UI
│   │   ├── app/                    # Ingest, Timeline Studio, Subtitle Grid, Blur Canvas
│   │   └── components/             # WaveSurfer, Konva Blur Box, OTIO Scrubber
│   ├── api/                        # FastAPI gateway + SSE progress endpoints
│   └── worker/                     # Temporal / Celery workers (Ingest, Render, Publish)
│
├── packages/
│   ├── core/                       # Shared Pydantic v2 models, R2/S3 client, DB (SQLAlchemy)
│   ├── perception/                 # PySceneDetect, decord, VLM indexer, syncstart, YAMNet, OCR
│   ├── audio_engine/               # Demucs, DeepFilterNet, Edge/Kokoro/ElevenLabs, FFmpeg mix
│   ├── vision_fx/                  # Ultralytics YOLO + ByteTrack redaction, Pillow/Playwright cards
│   ├── subtitles/                  # difflib alignment, pysubs2, 2-row ASS generator
│   ├── otio_engine/                # OpenTimelineIO compiler -> FFmpeg DAG + FCPXML/Resolve export
│   ├── agent_runtime/              # Native Tool Registry, Pydantic AI agents, Checkpoints, QA Critic
│   └── skills/                     # Niche editing playbooks (bodycam_doc.md, courtroom.md, etc.)
│
├── gpu_services/
│   └── modal_app.py                # Serverless GPU endpoints (WhisperX, Demucs, YOLO Blur, Qwen-VL)
│
└── tests/
    ├── unit/                       # OTIO invariants, 48kHz lock tests, difflib sync tests
    └── integration/                # End-to-end 30s fixture video render + FFprobe QA audit
```

---

## 7. End-to-End 10-Stage Execution Pipeline

1. **Stage 1 — Cloud Ingest & Proxy (`Ingest Worker`):** Downloads raw URL(s) + PDF reports directly in cloud → generates 480p proxy, scrubsheet, and `64k` audio in R2.
2. **Stage 2 — Multi-Cam Sync & Perception (`Perceiver Agent`):** Runs `syncstart` (if >1 camera), `PySceneDetect`, `Gladia/WhisperX` ASR, `YAMNet` audio events, `PaddleOCR`, and Stage-1 VLM captions → saves unified **Multimodal Timeline Index** in Postgres.
3. **Stage 3 — Editorial Planning (`Director Agent`):** Reads the Multimodal Index + `bodycam_doc.md` skill → calls `perception.inspect_frames` on candidate climax moments → calls `timeline.*` tools to construct `OTIO v1` (hook, dead-air trims, multi-cam switches, narrator script, J/L-cuts, freeze-callouts, chapter markers).
4. **Stage 4 — Privacy & Redaction Scan (`GPU Worker`):** Runs `Ultralytics YOLO + ByteTrack` on raw clips to pre-populate `redact.add_blur_region` proposals (bystander faces, license plates, corner channel watermarks).
5. **Stage 5 — Audio Prep & TTS (`Audio Worker`):** Runs `Demucs` on any music-flagged spans (vocals stem feeds the enhancer), the 2-Bus Speech Isolation pre-pass (`audio.enhance_speech`: neural denoise → EQ/comp → VAD+ASR speech gate → ambience bed → limiter; quantitative raw-vs-clean SNR receipt + 15 s A/B export), generates tight Brian TTS clips, and normalizes everything strictly to `48000 Hz stereo`.
6. **Stage 6 — Subtitle Alignment & Graphics (`Assembler Agent`):** Runs police-jargon correction + `difflib` alignment → builds 2-row speaker-colored `.ass` + renders `Pillow/Playwright` graphic cards.
7. **Stage 7 — Automated QA (`Critic Agent`):** Runs FFmpeg black/freeze/loudness checks + bounding-box collision checks + VLM spot-check on key frames.
8. **Stage 8 — Human-in-the-Loop Gate ("Director Studio" UI):** Workflow pauses cleanly. You review the proxy timeline, tweak any script line/subtitle/blur box in the browser, and click **Approve & Render** (only dirty segments re-render!).
9. **Stage 9 — Cached Master Render (`Render Worker`):** Compiles `.otio` → renders missing segment hashes → verifies identical `1080p/30fps/48kHz` streams → merges via `ffmpeg concat -c copy`.
10. **Stage 10 — Cloud-to-YouTube Draft (`Publish Worker`):** Uploads final MP4, `.srt`, chapters, and thumbnail straight to YouTube Studio as a **Private Draft** (plus optional `.otio` download for Resolve/Premiere).

---

## 8. Phased Engineering Roadmap (Milestones & Definition of Done)

### Milestone 1: Modular Core Engine + OTIO + Segment Caching (Local Docker)
* **Build:** Extract our Colab logic into `packages/{core,audio_engine,subtitles,vision_fx,otio_engine}`. Replace Python lists with **`OpenTimelineIO`** + hash-based `segment_cache` + PostgreSQL persistence.
* **Fix permanently (via unit tests):** `loudnorm` post-resample `48 kHz` lock, `-ss` + `-t` exact slicing, upright `Roboto-Bold.ttf` selection, `difflib` word-alignment, and 2-row ASS non-overlap.
* **Done when:** Running `uv run director render <url>` locally in Docker produces a full 1080p MP4 + `.otio`, and changing 1 narrator line re-renders in **<5 seconds** using cached chunks.

### Milestone 2: Native Agentic Tool Runtime + Durable Orchestration
* **Build:** Implement `packages/agent_runtime` (Typed Tool Registry, `Pydantic AI` Perceiver/Director/Critic agents, checkpoint/rollback) + `FastAPI` + `Temporal` (or `Celery/Redis`) + `Gladia` + `YouTube Draft Upload`.
* **Done when:** Giving the system only a raw bodycam URL triggers all stages autonomously, survives a simulated worker crash mid-render without re-running ASR/TTS, and passes `qa.audit_*` checks.

### Milestone 3: "Director Studio" Next.js Web UI
* **Build:** `apps/web` (`Next.js 16 + shadcn/ui + WaveSurfer.js + react-konva + TanStack Table`).
* **Done when:** You can paste a URL in the browser, watch SSE progress, scrub the 480p proxy, edit a subtitle/script line, draw a blur box with your mouse over a corner watermark, and publish a YouTube Private Draft without touching a terminal.

### Milestone 4: Multimodal Vision + $0 Compute Stack (ComputeBackend Port)
* **Build:** the `ComputeBackend` port (`vision_fx/compute`, port-and-adapter like `LLMBackend`/`TtsBackend`) with three adapters behind one `COMPUTE_BACKEND` env switch — `LocalOnnxCpuBackend` (DEFAULT, $0 zero card: `yolov8n-face.onnx` via onnxruntime, stride-3 sampling + constant-velocity tracking + linear interpolation, packaged YuNet fallback, on-demand Demucs only on music-flagged spans), `KaggleBatchGpuBackend` (OPTIONAL free cloud GPU, $0 zero card: `KGAT_*` bearer auth → `/api/v1/kernels/push` script kernels on a free Tesla T4 for heavy WhisperX / Demucs / YOLO sweeps, raw detections tracked host-side by the SAME tracker), `ModalGpuBackend` (DORMANT, wired behind `MODAL_TOKEN_ID`: sub-second serverless T4/L4/A10G bursts the day a card exists — Modal's card-verification fees are deliberately avoided). Plus `PySceneDetect` + VLM `Frames-on-Demand` + `syncstart` multi-cam switching + `Playwright` animated maps/cards.
* **Redaction is a 3-pass pipeline (issue #14, NOT static boxes):** Pass 0 `normalize_cfr` (strict 30fps CFR so detection frame index i == render frame index i); Pass 1 imgsz-960 / conf-0.22 YOLO sweep + BoT-SORT-style Global Motion Compensation (sparse LK flow on the same decode sweep) + `track_buffer=60` — whip-pans keep identities; Pass 2 whole-history trajectory repair (bridge gaps <=25 frames with +15% inflation, >25-frame holes SPLIT the window, 5-frame lead-in/out, One-Euro filter instead of EMA, envelope scale_w=1.35/scale_h=1.40 UNION raw box => padded >= raw by construction); Pass 3 per-frame feathered elliptical blur via rawvideo pipe + audio copy, `debug_overlay` mode, and the zero-leak gate: an integration test on a fast-pan clip proves **0 unblurred face frames** pixel-by-pixel. Worker seam: `REDACT_MODE=tracked` bakes the plan into a content-addressed redacted source; static regions stay for the studio canvas.
* **Plus:** license-plate redaction rides the SAME 3-pass law (second detector family: ml-debi yolov8n plate ONNX, SHA-pinned fetch; `merge_redaction_tracks` prefixes `plate_N` ids; one merged face+plate plan under one zero-leak audit) and `perception.audio_events` (YAMNet ONNX CPU, mel frontend baked into the graph: siren/gunshot/shout/glass/alarm spans persisted to `shots_json["audio_events"]`), speech-gated leveling (issue #15: RMS strictly on ASR speech spans, -12..-6 dBFS window, -1.5 dBTP alimiter), night-rescue + multi-cam exposure matching (issue #16: subtle gamma only, no stylization), `evidence_fact` middle-card (issue #17), YouTube 16:9 safe-zone + clean out-frame QA (issue #18: blackdetect head/tail + freezedetect tail), story-first retention scoring (issue #19: hook/acts/momentum/open-loops/dead-spans) + Studio NLE shortcuts.
* **Done when:** Multi-camera FOIA packets auto-sync, strip background copyrighted music (Demucs via Kaggle or local extra — **Kaggle lane PROVEN live 2026-11-18: real Eustis FOIA audio through a free T4 kernel, stems 244.69 s == source, vocals −21.1 dB mean vs music bed −31.4 dB; runner `scripts/demucs_stems_run.py`**), auto-blur bystander faces with persistent track IDs that survive pans (zero-leak gate GREEN), dialogue sits in -12..-6 dBFS without siren skew, dark footage is legible without stylization, and perform AI-directed punch-in zooms and freeze-callouts — at $0 compute burn with no payment method on file.

### Milestone 5: Cloud Deployment & Multi-Channel Hardening
* **Build:** Production deploy on `Hetzner` (or `Oracle Always Free`) + `Cloudflare R2` + `Modal` + `Sentry` + `Langfuse/Logfire` cost dashboards + multiple `SKILL.md` niche templates.
* **Done when:** Monthly cost per 15-min video is tracked transparently, failed jobs alert cleanly, and the platform runs reliably for real monetized channels.

---

## 9. Hard-Won Engineering Guardrails (Learned From Prototype)

Every bug discovered during our R&D phase becomes an automated CI test:
1. **FFmpeg `loudnorm` 192 kHz Trap:** Always place `-ar 48000 -ac 2` *after* `loudnorm` on every segment before `concat -c copy`, or mixed sample rates silently mute downstream narrator clips.
2. **FFmpeg Input Seeking Trap:** Never combine `-ss` before `-i` with `-to` after `-i`; always compute exact duration `dur = end - start` and pass `-ss <start> -t <dur>`.
3. **Wikimedia / FOIA Scraper Blocks:** Always pass a descriptive HTTP `User-Agent` header on cloud ingest.
4. **Why Volume-Based Silence Cutting Fails on Bodycams:** Highway traffic and cruiser A/C defeat volume thresholds (`auto-editor`); always cut dead air using **speech + audio-event + motion intervals**.
5. **Subtitle Sync Drift After Proofreading:** Never linearly interpolate text when words are edited; always align via `difflib.SequenceMatcher` against raw ASR word timestamps.
6. **Simultaneous Speaker Stacking:** Never clamp overlapping lines with forced minimums on a single row; assign distinct speakers to **Row 1 (`MarginV=46`)** and **Row 2 (`MarginV=96`)** so both remain legible without vertical snapping.
7. **Font Weight Matching:** Match exact upright filenames (`Roboto-Bold.ttf`), never substring `"Roboto-Bold"` (which matches `Roboto-BoldItalic.ttf`).
8. **Licensing Safety at Scale:** Avoid Remotion above small-team thresholds; use `FFmpeg + Pillow + Playwright` or Apache-2.0 `HyperFrames`.