"""Benchmark runs and their reports.

Two modes:
- recordings: each file is processed alone, frame by frame. Repeatable, shows what every
  stage costs and which plates come out.
- live: the full engine runs on the real cameras for a while. Shows what four cameras cost
  together, including how old a frame is by the time its result is ready.
"""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import onnxruntime as ort

from alpr_id import __version__
from alpr_id.bench.offline import run_recording
from alpr_id.config import AppConfig
from alpr_id.pipeline.engine import Engine, load_models

# total = work on one frame, including waiting for a model another camera is using.
# frame_age = how old the newest frame was when it was picked up.
# capture_to_result = the two together: from the frame leaving the decoder to its result.
STAGE_ORDER = (
    "decode", "convert", "preprocess", "detect", "track", "select", "ocr", "vote", "emit",
    "total", "frame_age", "capture_to_result",
)


def environment(config: AppConfig, providers: list[str]) -> dict[str, Any]:
    return {
        "alpr_id": __version__,
        "python": platform.python_version(),
        "machine": platform.node(),
        "processor": platform.processor(),
        "os": platform.platform(),
        "onnxruntime": ort.__version__,
        "providers": providers,
        "detector": str(config.detector.model_path),
        "ocr": str(config.ocr.model_path),
        "serialize_inference": config.runtime.serialize_inference,
    }


def benchmark_recordings(
    config: AppConfig,
    files: dict[str, Path],
    start: float = 0.0,
    end: float | None = None,
    crops_dir: Path | None = None,
) -> dict[str, Any]:
    """With `crops_dir`, the plate crop of every event is saved there for checking by eye."""
    detector, ocr = load_models(config)
    report: dict[str, Any] = {"mode": "recordings", "environment": environment(config, detector.providers), "cameras": {}}
    if crops_dir is not None:
        crops_dir.mkdir(parents=True, exist_ok=True)

    def save_crop(event, crop, _frame):
        if crops_dir is not None and crop is not None:
            name = f"{event.camera_id}_{event.timestamp[11:19].replace(':', '')}_{event.plate or 'EMPTY'}.jpg"
            cv2.imwrite(str(crops_dir / name), crop)
        return event

    for camera_id, path in files.items():
        camera = config.camera(camera_id)
        if camera is None:
            raise KeyError(f"camera {camera_id} is not in the config")
        result = run_recording(path, camera, config, detector, ocr, on_event=save_crop, start_seconds=start, max_seconds=end)
        report["cameras"][camera_id] = {
            "file": str(path),
            "frames_decoded": result.frames_decoded,
            "frames_processed": result.frames_processed,
            "process_fps": camera.process_fps,
            "latency_ms": result.latency,
            "events": [event.to_dict() for event in result.events],
        }
    return report


def benchmark_live(config: AppConfig, seconds: float, warmup: float = 10.0) -> dict[str, Any]:
    """Measure for `seconds` after `warmup`, so connecting and first-run costs are left out."""
    engine = Engine(config)
    engine.start()
    try:
        time.sleep(warmup)
        for worker in engine._workers.values():
            worker.timer.reset()
        time.sleep(seconds)
        cameras = engine.cameras()
        latency = {worker.camera.id: worker.timer.summary() for worker in engine._workers.values()}
        events = engine.store.query(limit=1000, since_epoch=engine._started_at)
        providers = engine.detector.providers
    finally:
        engine.stop()
    return {
        "mode": "live",
        "seconds": seconds,
        "environment": environment(config, providers),
        "cameras": {
            status["id"]: {
                "connected": status["connected"],
                "stream_fps": status["fps"],
                "processed_fps": status["processed_fps"],
                "process_fps_target": status["process_fps"],
                "reconnects": status["reconnects"],
                "latency_ms": latency.get(status["id"], {}),
            }
            for status in cameras
        },
        "events": events,
    }


def to_markdown(report: dict[str, Any]) -> str:
    env = report["environment"]
    lines = [
        f"# Benchmark alpr-id ({report['mode']})",
        "",
        f"- Mesin: {env['machine']}, {env['processor']}, {env['os']}",
        f"- Provider: {', '.join(env['providers'])} (onnxruntime {env['onnxruntime']})",
        f"- Detektor: `{env['detector']}`; OCR: `{env['ocr']}`",
    ]
    if report["mode"] == "live":
        lines.append(f"- Durasi: {report['seconds']:.0f} detik, semua kamera berjalan bersamaan")
    lines.append("")
    for camera_id, data in report["cameras"].items():
        if report["mode"] == "live":
            lines.append(
                f"## {camera_id}: stream {data['stream_fps']} fps, diproses {data['processed_fps']} fps "
                f"(target {data['process_fps_target']}), reconnect {data['reconnects']}"
            )
        else:
            lines.append(f"## {camera_id}: {data['frames_processed']} frame diproses dari {data['frames_decoded']} frame")
        lines += ["", "| Tahap | n | rata-rata ms | p50 ms | p95 ms | maks ms |", "|---|---|---|---|---|---|"]
        latency = data["latency_ms"]
        for stage in [s for s in STAGE_ORDER if s in latency] + [s for s in latency if s not in STAGE_ORDER]:
            stats = latency[stage]
            lines.append(f"| {stage} | {stats['count']} | {stats['mean_ms']} | {stats['p50_ms']} | {stats['p95_ms']} | {stats['max_ms']} |")
        lines.append("")
        for event in data.get("events", []):
            lines.append(f"- event: `{event['plate_formatted']}` confidence {event['confidence']}, {event['reads']} bacaan, format_valid={event['format_valid']}")
        if data.get("events"):
            lines.append("")
    if report["mode"] == "live":
        lines.append(f"Event selama pengukuran: {len(report['events'])}")
        for event in report["events"]:
            lines.append(f"- {event['timestamp']} {event['camera_id']}: `{event['plate_formatted']}` confidence {event['confidence']}")
    return "\n".join(lines) + "\n"


def save_report(report: dict[str, Any], out_dir: str | Path = "data/benchmarks") -> tuple[Path, Path]:
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%Y%m%d-%H%M%S}-{report['mode']}"
    json_path, md_path = folder / f"{stem}.json", folder / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    return json_path, md_path
