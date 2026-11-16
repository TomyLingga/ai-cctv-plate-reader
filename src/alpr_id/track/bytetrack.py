"""ByteTrack per camera, fed with detections from our own detector."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from alpr_id.config import TrackingConfig
from alpr_id.detect.onnx_detector import Detection


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection over union of two x1, y1, x2, y2 boxes."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class TrackedPlate:
    track_id: int
    box: tuple[float, float, float, float]
    confidence: float

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


class CameraTracker:
    """One tracker instance per camera. Track ids start at 0 and are unique per camera only."""

    def __init__(self, config: TrackingConfig, frame_rate: float) -> None:
        self._tracker = ByteTrackTracker(
            # The library counts this in frames of a 30 fps video, i.e. thirtieths of a second.
            lost_track_buffer=max(int(round(config.lost_track_s * 30)), 1),
            frame_rate=frame_rate,
            track_activation_threshold=config.track_activation_threshold,
            minimum_consecutive_frames=config.minimum_consecutive_frames,
            high_conf_det_threshold=config.high_conf_det_threshold,
        )

    def update(self, detections: list[Detection], timestamp: float) -> list[TrackedPlate]:
        """Confirmed tracks visible in this frame. `timestamp` is the capture time in seconds."""
        if detections:
            observed = sv.Detections(
                xyxy=np.array([d.box for d in detections], dtype=np.float32),
                confidence=np.array([d.confidence for d in detections], dtype=np.float32),
                class_id=np.zeros(len(detections), dtype=int),
            )
        else:
            observed = sv.Detections.empty()
        tracked = self._tracker.update(observed, timestamp=timestamp)
        if tracked.tracker_id is None or len(tracked) == 0:
            return []
        confidences = tracked.confidence if tracked.confidence is not None else np.ones(len(tracked))
        return [
            TrackedPlate(int(track_id), tuple(float(v) for v in box), float(conf))
            for box, track_id, conf in zip(tracked.xyxy, tracked.tracker_id, confidences, strict=True)
            if track_id >= 0  # -1 marks a track that is not confirmed yet
        ]
