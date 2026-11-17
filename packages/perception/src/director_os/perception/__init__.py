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
