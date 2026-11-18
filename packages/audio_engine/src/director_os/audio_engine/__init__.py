"""DirectorOS audio engine — loudness law, pacing, ducking, TTS, speech
isolation 2-bus mix (PLAN §3.3D, §4.5)."""

from director_os.audio_engine.ab_export import best_speech_window, export_ab_pair
from director_os.audio_engine.enhance import (
    EnhancerError,
    EnhancerResult,
    EnhancerUnavailableError,
    SpeechEnhancer,
    resolve_enhancer,
)
from director_os.audio_engine.tts import (
    EdgeTtsBackend,
    ElevenLabsTtsBackend,
    TtsBackend,
    TtsError,
    TtsResult,
    build_tts_backend,
    generate_tts,
    generate_tts_cached,
    tts_cache_key,
)
from director_os.audio_engine.two_bus import (
    TwoBusMixError,
    apply_gate_inplace,
    build_two_bus_mix_cmd,
    bus1_tone_chain,
    bus2_chain,
    duck_envelope,
    master_chain,
    segment_fade_chain,
)

__all__ = [
    "EdgeTtsBackend",
    "ElevenLabsTtsBackend",
    "EnhancerError",
    "EnhancerResult",
    "EnhancerUnavailableError",
    "SpeechEnhancer",
    "TtsBackend",
    "TtsError",
    "TtsResult",
    "TwoBusMixError",
    "apply_gate_inplace",
    "best_speech_window",
    "build_tts_backend",
    "build_two_bus_mix_cmd",
    "bus1_tone_chain",
    "bus2_chain",
    "duck_envelope",
    "export_ab_pair",
    "generate_tts",
    "generate_tts_cached",
    "master_chain",
    "resolve_enhancer",
    "segment_fade_chain",
    "tts_cache_key",
]
