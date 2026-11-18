"""Run one recording through the camera pipeline with and without the camera's focus area.

Prints, for each of the two runs: how many processed frames had a plate in view, how many
had a plate good enough for /plate/{n}, the events, and the detector time per frame. The
recording is processed at the camera's process_fps, as the live service does.

    uv run --frozen --extra cpu python scripts/compare_focus.py data/samples/rec-20261113-b/cam-4.mkv cam-4 --out out.json

The camera's `focus` comes from config/app.yaml, or from --focus x1,y1,x2,y2.

--detector FILE runs the same with another detector than the one in the config. That is how a
candidate detector is tried on a recording before it is installed: tracking, waiting for the
truck to stop, OCR and voting are the service's own.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import av

from alpr_id.config import load_config
from alpr_id.logging_setup import configure_logging
from alpr_id.pipeline.camera_pipeline import CameraPipeline
from alpr_id.pipeline.engine import load_models
from alpr_id.timing import StageTimer


def run(path: Path, camera, config, detector, ocr, max_seconds: float | None) -> dict:
    timer = StageTimer(keep=1_000_000)
    pipeline = CameraPipeline(camera, config, detector, ocr, timer)
    interval = 1.0 / camera.process_fps
    next_due, processed, in_view, reportable = 0.0, 0, 0, 0
    events = []
    base = time.time()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            t = float(frame.pts * stream.time_base) if frame.pts is not None else index / 20.0
            if max_seconds is not None and t > max_seconds:
                break
            if t + 1e-6 < next_due:
                continue
            next_due = max(next_due + interval, t)
            outcome = pipeline.process(frame.to_ndarray(format="bgr24"), t, base + t)
            processed += 1
            in_view += pipeline.in_view(t) > 0
            reportable += pipeline.current(t) is not None
            events.extend((round(t, 1), e) for e in outcome.events)
    events.extend((None, e) for e in pipeline.flush())
    latency = timer.summary()
    return {
        "focus": list(camera.focus) if camera.focus else None,
        "frames": processed,
        "frames_plate_in_view": in_view,
        "frames_plate_reportable": reportable,
        "detect_ms_p50": round(latency.get("detect", {}).get("p50_ms", 0.0), 1),
        "preprocess_ms_p50": round(latency.get("preprocess", {}).get("p50_ms", 0.0), 1),
        "events": [
            {"t": t, "plate": e.plate, "confidence": round(e.confidence, 3), "stopped": e.stopped, "bbox": list(e.bbox)}
            for t, e in events
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording", type=Path)
    parser.add_argument("camera_id")
    parser.add_argument("--focus", help="x1,y1,x2,y2 as fractions; default: the camera's focus in the config")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--only", choices=("wide_only", "wide_and_focus"), help="run one of the two, to run them side by side")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--detector", type=Path, help="detector ONNX file to use instead of detector.model_path")
    args = parser.parse_args()

    configure_logging(log_dir=None, level="WARNING")
    config = load_config()
    if args.detector:
        config = config.model_copy(update={"detector": config.detector.model_copy(update={"model_path": args.detector})})
    camera = config.camera(args.camera_id)
    if camera is None:
        raise SystemExit(f"unknown camera {args.camera_id}")
    focus = tuple(float(v) for v in args.focus.split(",")) if args.focus else camera.focus
    if focus is None:
        raise SystemExit("the camera has no focus area; give one with --focus")
    detector, ocr = load_models(config)

    result = {"recording": str(args.recording), "camera": camera.id, "detector": str(config.detector.model_path)}
    for name, variant in (("wide_only", camera.model_copy(update={"focus": None})), ("wide_and_focus", camera.model_copy(update={"focus": focus}))):
        if args.only and name != args.only:
            continue
        result[name] = run(args.recording, variant, config, detector, ocr, args.max_seconds)
        r = result[name]
        print(f"{camera.id} {name}: {r['frames']} frames, plate in view {r['frames_plate_in_view']}, "
              f"reportable {r['frames_plate_reportable']}, {len(r['events'])} events, detect p50 {r['detect_ms_p50']} ms", flush=True)
    if args.out:
        args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
