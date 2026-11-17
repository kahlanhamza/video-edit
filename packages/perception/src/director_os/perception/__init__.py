"""DirectorOS perception layer — populated in M2 (see README.md)."""

from director_os.perception.faces import (  # noqa: F401
    FaceDetection,
    FaceSweep,
    FaceSweepStats,
    YuNetFaceDetector,
    cluster_to_regions,
    detect_faces,
    make_detector,
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
