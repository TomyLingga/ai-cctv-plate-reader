"""Run the camera pipeline over a recording, frame by frame, without real-time pressure.

Gives repeatable plate results and per-stage latency for one file. Frames are taken at the
camera's `process_fps`, as the live worker would.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import av

from alpr_id.config import AppConfig, CameraConfig
from alpr_id.detect.onnx_detector import PlateDetector
from alpr_id.ocr.recognizer import PlateOcr
from alpr_id.pipeline.camera_pipeline import CameraPipeline, EventSink, ReadSink
from alpr_id.pipeline.events import PlateEvent
from alpr_id.timing import StageTimer


@dataclass
class RecordingResult:
    path: str
    frames_decoded: int = 0
    frames_processed: int = 0
    duration_s: float = 0.0
    events: list[PlateEvent] = field(default_factory=list)
    latency: dict[str, dict[str, float]] = field(default_factory=dict)


def run_recording(
    path: str | Path,
    camera: CameraConfig,
    config: AppConfig,
    detector: PlateDetector,
    ocr: PlateOcr,
    on_event: EventSink | None = None,
    on_read: ReadSink | None = None,
    start_epoch: float | None = None,
    start_seconds: float = 0.0,
    max_seconds: float | None = None,
) -> RecordingResult:
    """Process `path` from `start_seconds` (snapped back to a keyframe) to `max_seconds`."""
    timer = StageTimer(keep=1_000_000)
    pipeline = CameraPipeline(camera, config, detector, ocr, timer, on_event=on_event, on_read=on_read)
    result = RecordingResult(path=str(path))
    base_epoch = start_epoch if start_epoch is not None else time.time()
    interval = 1.0 / camera.process_fps
    next_due = 0.0

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if start_seconds > 0:
            container.seek(int(start_seconds * av.time_base))
            next_due = start_seconds
        decode_started = time.perf_counter()
        for index, frame in enumerate(container.decode(stream)):
            timer.add("decode", (time.perf_counter() - decode_started) * 1000.0)
            t = float(frame.pts * stream.time_base) if frame.pts is not None else index / 20.0
            result.frames_decoded += 1
            result.duration_s = t
            if max_seconds is not None and t > max_seconds:
                break
            if t + 1e-6 >= next_due:
                next_due = max(next_due + interval, t)
                started = time.perf_counter()
                with timer.stage("convert"):
                    image = frame.to_ndarray(format="bgr24")
                outcome = pipeline.process(image, t, base_epoch + t)
                timer.add("total", (time.perf_counter() - started) * 1000.0)
                result.frames_processed += 1
                result.events.extend(outcome.events)
            decode_started = time.perf_counter()

    result.events.extend(pipeline.flush())
    result.latency = timer.summary()
    return result
