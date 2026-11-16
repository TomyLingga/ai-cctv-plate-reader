"""Compare detector ONNX files through the runtime's own code on labelled frames.

Ultralytics' validation measures the .pt file with its own pre- and post-processing. The
service loads the ONNX file with alpr_id.detect.PlateDetector, so this script is the check
that counts before a model is installed: same letterbox, same thresholds, same decoding.

For every model it reports, at the service's confidence threshold:
  found    labelled plates that got a box (IoU of at least 0.5)
  missed   labelled plates without one
  extra    boxes on something that has no label (a false alarm, or a plate the labels lack)
and the time one frame takes on this machine. Crops of the missed and the extra boxes go to
--review so a person can look at them.

Usage (from the project root, runtime environment):
    uv run --frozen --extra cpu python training/detector/compare_runtime.py --data data/dataset/detector \
        --split test_cctv --model current=models/detector/plat.onnx --model yolo26n=models/detector/candidates/yolo26n.onnx
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import cv2
import onnxruntime as ort

from alpr_id.config import load_config
from alpr_id.detect.onnx_detector import PlateDetector
from alpr_id.detect.providers import resolve_providers
from alpr_id.logging_setup import configure_logging
from alpr_id.timing import clock


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def save_crop(folder: Path, name: str, image, box: tuple[float, float, float, float]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = box
    margin_x, margin_y = (x2 - x1) * 1.5, (y2 - y1) * 1.5
    top, bottom = max(int(y1 - margin_y), 0), min(int(y2 + margin_y), image.shape[0])
    left, right = max(int(x1 - margin_x), 0), min(int(x2 + margin_x), image.shape[1])
    view = image[top:bottom, left:right].copy()
    cv2.rectangle(view, (int(x1) - left, int(y1) - top), (int(x2) - left, int(y2) - top), (0, 255, 0), 1)
    cv2.imwrite(str(folder / f"{name}.jpg"), view)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="dataset folder of build_dataset.py")
    parser.add_argument("--split", default="test_cctv")
    parser.add_argument("--model", action="append", required=True, metavar="NAME=FILE")
    parser.add_argument("--conf", type=float, default=None, help="default: detector.conf_threshold of the service")
    parser.add_argument("--review", type=Path, default=None, help="folder for crops of missed and extra boxes")
    parser.add_argument("--out", type=Path, default=None, help="write the numbers as JSON here")
    args = parser.parse_args()

    configure_logging(log_dir=None, level="ERROR")
    config = load_config()
    conf = args.conf if args.conf is not None else config.detector.conf_threshold
    images = sorted((args.data / "images" / args.split).glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images in {args.data / 'images' / args.split}")
    results = {}
    for item in args.model:
        name, _, path = item.partition("=")
        detector = PlateDetector(ort.InferenceSession(path, providers=resolve_providers(config.runtime)),
                                 conf_threshold=conf, iou_threshold=config.detector.iou_threshold)
        detector.detect(cv2.imread(str(images[0])))  # the first call pays for start-up
        found = missed = extra = labelled = 0
        times, scores = [], []
        for path_image in images:
            image = cv2.imread(str(path_image))
            height, width = image.shape[:2]
            truth = []
            for line in (args.data / "labels" / args.split / (path_image.stem + ".txt")).read_text(encoding="utf-8").splitlines():
                _, x, y, w, h = (float(v) for v in line.split())
                truth.append(((x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height))
            started = clock()
            boxes = detector.detect(image)
            times.append((clock() - started) * 1000)
            labelled += len(truth)
            unused = list(truth)
            for index, box in enumerate(sorted(boxes, key=lambda b: -b.confidence)):
                best = max(unused, key=lambda t: iou(box.box, t), default=None)
                if best is not None and iou(box.box, best) >= 0.5:
                    unused.remove(best)
                    found += 1
                    scores.append(box.confidence)
                else:
                    extra += 1
                    if args.review:
                        save_crop(args.review / name / "extra", f"{path_image.stem}_{index}_{box.confidence:.2f}", image, box.box)
            missed += len(unused)
            if args.review:
                for index, box in enumerate(unused):
                    save_crop(args.review / name / "missed", f"{path_image.stem}_{index}", image, box)
        results[name] = {
            "model": path, "provider": detector.providers[0], "confidence_threshold": conf, "frames": len(images),
            "labelled": labelled, "found": found, "missed": missed, "extra": extra,
            "mean_confidence_found": round(statistics.fmean(scores), 4) if scores else None,
            "ms_per_frame_median": round(statistics.median(times), 2),
            "ms_per_frame_p95": round(sorted(times)[int(len(times) * 0.95) - 1], 2),
        }
        r = results[name]
        print(f"{name:<12} found {found}/{labelled} ({found / max(labelled, 1):.1%}), missed {missed}, extra {extra}, "
              f"mean confidence {r['mean_confidence_found']}, {r['ms_per_frame_median']} ms per frame (p95 {r['ms_per_frame_p95']}) on {r['provider']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
