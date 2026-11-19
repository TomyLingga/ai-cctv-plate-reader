"""Scan a long recording: what the detector finds and the OCR reads in it, key frame by key frame.

For the question a few clips cannot answer: how large a plate has to be in the picture (how close
the truck has to stand) before it is found and before it is read. Every --every seconds one key
frame of a camera's recording goes through the two views the service gives its detector (the ROI
shrunk to the detector input, the focus area at camera resolution), for every detector named with
--model. Each box found, down to --propose, becomes one line of the CSV with its size, its score
in each view, and what the OCR reads from it.

    uv run --frozen --extra cpu python scripts/scan_recording.py data/samples/rec-long-20261118 --cameras 3 \\
        --model baru=models/detector/candidates/yolov8n-20261118b.onnx --model lama=models/detector/plat.onnx \\
        --out data/benchmarks/scan-20261118/cam-3.csv

The recording is the folder scripts/record_long.py writes: cam-<n>/<yyyymmdd-HHMMSS>.mkv, named
after the server clock at the first packet. Only key frames are decoded.

Run it in the CPU environment on the laptop. In the GPU environment (RTX 5050) four of eight
scans running side by side read rubbish from every plate for the whole recording, although
their OCR sessions were on the CPU and only the detectors on the GPU (2026-11-19); the same
files read correctly when scanned again. Everything here therefore runs on the CPU, and every
--check-every-th reading is made a second time by a second OCR session: when the two disagree
more than once in ten the scan stops instead of writing rubbish.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import av
import onnxruntime as ort

from alpr_id.capture.roi import PixelRect, clamp_box, roi_to_pixels
from alpr_id.config import load_config
from alpr_id.detect.onnx_detector import PlateDetector
from alpr_id.detect.providers import resolve_providers
from alpr_id.logging_setup import configure_logging
from alpr_id.ocr.recognizer import PlateOcr
from alpr_id.pipeline.camera_pipeline import EDGE_MARGIN_PX
from alpr_id.postprocess.correction import normalize_plate
from alpr_id.track.bytetrack import box_iou

FIELDS = ["camera", "time", "epoch", "detector", "x1", "y1", "x2", "y2", "width", "score_roi", "score_focus",
          "text", "ocr_confidence", "format_valid", "region_known"]


def quiet(threads: int) -> ort.SessionOptions:
    """Few threads that sleep between calls. One scan runs per camera side by side, and ONNX Runtime's
    default (a thread per core, each spinning while it waits) made four of them fight over the CPU:
    six cores busy per process and a 30 minute file took two and a half minutes."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return options


def look(detector: PlateDetector, frame, view: PixelRect) -> list:
    """Boxes of one view in frame coordinates, without those cut by the view's edge (the service drops them too)."""
    kept = []
    for found in detector.detect(view.crop(frame)):
        if (found.x1 <= EDGE_MARGIN_PX or found.y1 <= EDGE_MARGIN_PX
                or found.x2 >= view.width - EDGE_MARGIN_PX or found.y2 >= view.height - EDGE_MARGIN_PX):
            continue
        kept.append(found.shifted(view.x1, view.y1))
    return kept


def plates_of(detector: PlateDetector, frame, roi: PixelRect, focus: PixelRect | None) -> list[dict]:
    """One entry per plate: the box of the view that is surer of it, and the score in each view (0 when not found there)."""
    plates = [{"box": d.box, "score_roi": d.confidence, "score_focus": 0.0} for d in look(detector, frame, roi)]
    for found in look(detector, frame, focus) if focus is not None else []:
        same = next((p for p in plates if box_iou(p["box"], found.box) >= 0.5), None)
        if same is None:
            plates.append({"box": found.box, "score_roi": 0.0, "score_focus": found.confidence})
        else:
            same["score_focus"] = found.confidence
            if found.confidence > same["score_roi"]:
                same["box"] = found.box
    return plates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording", type=Path)
    parser.add_argument("--cameras", nargs="+", required=True, help="camera numbers, as in the folder names cam-<n>")
    parser.add_argument("--model", action="append", required=True, metavar="NAME=FILE")
    parser.add_argument("--every", type=float, default=5.0, help="seconds between the key frames looked at")
    parser.add_argument("--propose", type=float, default=0.15, help="detector confidence from which a box is written down")
    parser.add_argument("--read-from", type=float, default=0.35, help="detector confidence from which the OCR reads a box")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--part", default="1/1", metavar="I/N", help="take every N-th file starting with the I-th, to run N scans side by side")
    parser.add_argument("--check-every", type=int, default=10, help="every how many readings the second OCR session reads too")
    parser.add_argument("--gpu", action="store_true", help="detectors on the providers of the config instead of the CPU (see above)")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads per detector session; one scan also needs about 0.6 GB of memory")
    args = parser.parse_args()
    part, parts = (int(v) for v in args.part.split("/"))

    configure_logging(log_dir=None, level="ERROR")
    config = load_config()
    providers = resolve_providers(config.runtime) if args.gpu else ["CPUExecutionProvider"]
    detectors = {}
    for item in args.model:
        name, _, path = item.partition("=")
        detectors[name] = PlateDetector(ort.InferenceSession(path, sess_options=quiet(args.threads), providers=providers),
                                        conf_threshold=args.propose, iou_threshold=config.detector.iou_threshold)
    ocr = PlateOcr(config.ocr.model_path, config.ocr.plate_config_path, ["CPUExecutionProvider"])
    # PlateOcr has no way to pass session options; swap its session for a quiet one of the same model.
    ocr._session = ort.InferenceSession(str(config.ocr.model_path), sess_options=quiet(4), providers=["CPUExecutionProvider"])
    witness = PlateOcr(config.ocr.model_path, config.ocr.plate_config_path, ["CPUExecutionProvider"])
    witness._session = ort.InferenceSession(str(config.ocr.model_path), sess_options=quiet(2), providers=["CPUExecutionProvider"])
    reads = checked = disagreed = 0
    cameras = {camera.id: camera for camera in config.cameras}
    print("detectors on", next(iter(detectors.values())).providers[0], "| OCR on", ocr.providers[0], flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames = rows = 0
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for number in args.cameras:
            camera = cameras[f"cam-{number}"]
            for path in sorted((args.recording / f"cam-{number}").glob("*.mkv"))[part - 1::parts]:
                started = datetime.strptime(path.stem, "%Y%m%d-%H%M%S")
                last, first_pts = -1e9, None
                try:
                    container = av.open(str(path))
                except av.error.FFmpegError as exc:
                    print(f"{path.name}: cannot be opened ({type(exc).__name__})", flush=True)
                    continue
                with container:
                    stream = container.streams.video[0]
                    stream.codec_context.skip_frame = "NONKEY"
                    try:
                        for frame in container.decode(stream):
                            if frame.pts is None:
                                continue
                            first_pts = frame.pts if first_pts is None else first_pts
                            moment = float((frame.pts - first_pts) * stream.time_base)
                            if moment - last < args.every:
                                continue
                            last = moment
                            frames += 1
                            image = frame.to_ndarray(format="bgr24")
                            height, width = image.shape[:2]
                            roi = roi_to_pixels(camera.roi, width, height)
                            focus = roi_to_pixels(camera.focus, width, height) if camera.focus is not None else None
                            when = started + timedelta(seconds=moment)
                            for name, detector in detectors.items():
                                for plate in plates_of(detector, image, roi, focus):
                                    box = plate["box"]
                                    line = {"camera": camera.id, "time": when.isoformat(timespec="seconds"), "epoch": round(when.timestamp(), 1),
                                            "detector": name, "x1": round(box[0]), "y1": round(box[1]), "x2": round(box[2]), "y2": round(box[3]),
                                            "width": round(box[2] - box[0]), "score_roi": round(plate["score_roi"], 3),
                                            "score_focus": round(plate["score_focus"], 3)}
                                    area = clamp_box(box, width, height, config.ocr.crop_padding)
                                    if area is not None and max(plate["score_roi"], plate["score_focus"]) >= args.read_from:
                                        crop = area.crop(image)
                                        if config.ocr.crop_bottom_ratio > 0:  # the service cuts the expiry date off the same way
                                            crop = crop[: max(int(round(crop.shape[0] * (1.0 - config.ocr.crop_bottom_ratio))), 2)]
                                        read = ocr.read([crop])[0]
                                        reads += 1
                                        if reads % args.check_every == 0:
                                            checked += 1
                                            disagreed += witness.read([crop])[0].text != read.text
                                            if checked >= 20 and disagreed > checked / 10:
                                                raise SystemExit(f"the two OCR sessions disagree on {disagreed} of {checked} crops: "
                                                                 "one of them reads rubbish, nothing of this scan can be used")
                                        text = normalize_plate(read.text)
                                        line.update(text=text.plate, ocr_confidence=round(read.confidence, 3),
                                                    format_valid=int(text.format_valid), region_known=int(text.region_known))
                                    writer.writerow(line)
                                    rows += 1
                    except av.error.FFmpegError as exc:
                        print(f"{path.name}: stopped at {last:.0f} s ({type(exc).__name__})", flush=True)
                handle.flush()
                print(f"cam-{number} {path.stem}: {frames} frames, {rows} boxes so far", flush=True)
    print(f"done: {frames} frames, {rows} boxes, {reads} readings of which {checked} checked and {disagreed} disagreed -> {args.out}")


if __name__ == "__main__":
    main()
