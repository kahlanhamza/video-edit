"""DirectorOS perception layer — M2 ASR/shots + M4 vision stack (see README.md)."""

from director_os.perception.faces import (  # noqa: F401
    FaceDetection,
    FaceSweep,
    FaceSweepStats,
    YuNetFaceDetector,
    cluster_to_regions,
    detect_faces,
    make_detector,
)
from director_os.perception.frame_quality import (  # noqa: F401
    FrameQualityReport,
    FrameScore,
    black_frame_probe,
    score_frame_quality,
)
from director_os.perception.modal_redact import (  # noqa: F401
    ModalRedactError,
    TrackedObject,
    TrackFrame,
    merge_with_static,
    modal_configured,
    parse_tracks_payload,
    redact_video_modal,
    tracks_to_regions,
)
from director_os.perception.sync import (  # noqa: F401
    MultiCamSync,
    SyncError,
    SyncResult,
    cross_correlate,
    extract_audio,
    load_samples,
    sync_multicam,
    sync_pair,
)
from director_os.perception.timestamp_ocr import (  # noqa: F401
    BodycamStamp,
    TimestampOcrError,
    extract_frame,
    parse_timestamp_text,
    read_bodycam_timestamp,
)
from director_os.perception.vision import (  # noqa: F401
    FakeVisionClient,
    FrameRef,
    GeminiVisionClient,
    SceneInsight,
    VisionAdapter,
    VisionClient,
    VisionError,
    build_vision_client,
)
from director_os.perception.vision_index import (  # noqa: F401
    ContactSheet,
    VisionIndexError,
    build_contact_sheets,
    frames_on_demand,
    index_timeline,
    make_vision_clients,
    qa_placement,
    suggest_fod_timestamps,
)
