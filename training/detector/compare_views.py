"""Which views of a camera frame let the detector find the plates in it.

The service gives the detector two views of a frame: the camera's ROI shrunk to the detector
input, and the focus area at camera resolution. The old service (ai_cctv) did something else:
`ZOOM_FACTOR = 2.5` cut the middle of the frame, 1/2.5 of its width and height, and gave the
detector only that. This script puts both in front of the same frames and counts, for every
combination of views, the labelled plates that get a box strong enough to start a track.

The frames are the event frames of a dataset split, whole, as the service saved them; the
plates are the labels of that split's ROI pictures moved back into frame coordinates. A box
that touches the edge of its view is dropped, as the service drops it: it is half a plate.

Usage (from the project root, runtime environment):
    uv run --frozen --extra cpu python training/detector/compare_views.py --data data/dataset/detector-20261118b \
        --events data/rog-20261118 --zoom 2.5 --model models/detector/candidates/yolov8n-20261118b.onnx
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import cv2
import onnxruntime as ort

from alpr_id.capture.roi import PixelRect, roi_to_pixels
from alpr_id.config import load_config
from alpr_id.detect.onnx_detector import PlateDetector
from alpr_id.detect.providers import resolve_providers
from alpr_id.logging_setup import configure_logging
from alpr_id.pipeline.camera_pipeline import EDGE_MARGIN_PX
from alpr_id.track.bytetrack import box_iou

NIGHT_FROM, NIGHT_UNTIL = 18, 6  # hours of the server clock, as in build_dataset.py


def zoom_view(width: int, height: int, factor: float) -> PixelRect:
    """The middle of the frame, 1/factor of its width and height: what ai_cctv's ZOOM_FACTOR cut out."""
    view_w, view_h = int(width / factor), int(height / factor)
    x1, y1 = (width - view_w) // 2, (height - view_h) // 2
    return PixelRect(x1, y1, x1 + view_w, y1 + view_h)


def look(detector: PlateDetector, frame, view: PixelRect) -> list:
    """Boxes the detector gives in one view, in frame coordinates, without those cut by the view's edge."""
    kept = []
    for found in detector.detect(view.crop(frame)):
        if (found.x1 <= EDGE_MARGIN_PX or found.y1 <= EDGE_MARGIN_PX
                or found.x2 >= view.width - EDGE_MARGIN_PX or found.y2 >= view.height - EDGE_MARGIN_PX):
            continue
        kept.append(found.shifted(view.x1, view.y1))
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="dataset folder of build_dataset.py")
    parser.add_argument("--split", default="test_cctv")
    parser.add_argument("--events", type=Path, required=True, help="folder with media/<date>/<camera>/*_frame.jpg")
    parser.add_argument("--model", type=Path, default=None, help="default: detector.model_path of the config")
    parser.add_argument("--zoom", type=float, nargs="+", default=[2.5])
    parser.add_argument("--out", type=Path, default=None, help="write the numbers as JSON here")
    parser.add_argument("--review", type=Path, default=None,
                        help="folder for crops of strong boxes that are on no labelled plate and that no other view has")
    args = parser.parse_args()

    configure_logging(log_dir=None, level="ERROR")
    config = load_config()
    model = args.model or config.detector.model_path
    detector = PlateDetector(ort.InferenceSession(str(model), providers=resolve_providers(config.runtime)),
                             conf_threshold=config.detector.conf_threshold, iou_threshold=config.detector.iou_threshold)
    strong_from = config.tracking.track_activation_threshold
    cameras = {camera.id: camera for camera in config.cameras}
    names = ["roi", "focus", *(f"zoom {factor:g}" for factor in args.zoom)]

    # plate -> for every view: the best confidence of a box on it, 0 when there is none
    plates: list[dict] = []
    alone = dict.fromkeys(names, 0)  # strong boxes on no label that only this view has: a plate the labels lack, or a false alarm
    for picture in sorted((args.data / "images" / args.split).glob("cctv_ev_*_w.jpg")):
        _, _, date, camera_id, clock, tag, _ = picture.stem.split("_")
        frame = cv2.imread(str(args.events / "media" / date / camera_id / f"{clock}_{tag}_frame.jpg"))
        camera = cameras.get(camera_id)
        if frame is None or camera is None:
            continue
        height, width = frame.shape[:2]
        roi = roi_to_pixels(camera.roi, width, height)
        views = {"roi": roi}
        if camera.focus is not None:
            views["focus"] = roi_to_pixels(camera.focus, width, height)
        for factor in args.zoom:
            views[f"zoom {factor:g}"] = zoom_view(width, height, factor)
        found = {name: look(detector, frame, view) for name, view in views.items()}
        night = int(clock[:2]) >= NIGHT_FROM or int(clock[:2]) < NIGHT_UNTIL
        labelled = []
        for line in (args.data / "labels" / args.split / f"{picture.stem}.txt").read_text(encoding="utf-8").splitlines():
            _, x, y, w, h = (float(v) for v in line.split())
            box = (roi.x1 + (x - w / 2) * roi.width, roi.y1 + (y - h / 2) * roi.height,
                   roi.x1 + (x + w / 2) * roi.width, roi.y1 + (y + h / 2) * roi.height)
            labelled.append(box)
            plate = {"picture": picture.stem, "night": night, "width": box[2] - box[0], "scores": {}, "inside": {}}
            for name, view in views.items():
                plate["inside"][name] = box[0] >= view.x1 and box[1] >= view.y1 and box[2] <= view.x2 and box[3] <= view.y2
                plate["scores"][name] = max((d.confidence for d in found[name] if box_iou(d.box, box) >= 0.5), default=0.0)
            plates.append(plate)
        for name, boxes in found.items():
            others = [d.box for other, more in found.items() if other != name for d in more]
            for index, detection in enumerate(boxes):
                if detection.confidence < strong_from or any(box_iou(detection.box, b) >= 0.3 for b in [*labelled, *others]):
                    continue
                alone[name] += 1
                if args.review:
                    folder = args.review / name.replace(" ", "")
                    folder.mkdir(parents=True, exist_ok=True)
                    x1, y1, x2, y2 = (int(v) for v in detection.box)
                    pad_x, pad_y = (x2 - x1) * 2, (y2 - y1) * 2
                    crop = frame[max(y1 - pad_y, 0):y2 + pad_y, max(x1 - pad_x, 0):x2 + pad_x].copy()
                    cv2.rectangle(crop, (x1 - max(x1 - pad_x, 0), y1 - max(y1 - pad_y, 0)),
                                  (x2 - max(x1 - pad_x, 0), y2 - max(y1 - pad_y, 0)), (0, 255, 0), 1)
                    cv2.imwrite(str(folder / f"{picture.stem}_{index}_{detection.confidence:.2f}_w{x2 - x1}.jpg"), crop)

    def count(views: tuple[str, ...], group: list[dict]) -> int:
        return sum(1 for plate in group if max(plate["scores"].get(name, 0.0) for name in views) >= strong_from)

    groups = {"all": plates, "day": [p for p in plates if not p["night"]], "night": [p for p in plates if p["night"]]}
    print(f"{model}: {len(plates)} plates ({len(groups['day'])} day, {len(groups['night'])} night), "
          f"a plate counts with a box of confidence {strong_from} or more")
    print("plates lying whole inside a view: " + ", ".join(
        f"{name} {sum(1 for p in plates if p['inside'].get(name))}" for name in names))
    print("strong boxes on no label that only this view has: " + ", ".join(f"{name} {n}" for name, n in alone.items()))
    results = {}
    sets = [combo for size in range(1, len(names) + 1) for combo in combinations(names, size)]
    print(f"{'views':<28} {'all':>5} {'day':>5} {'night':>5}")
    for views in sets:
        row = {group: count(views, members) for group, members in groups.items()}
        results[" + ".join(views)] = row
        print(f"{' + '.join(views):<28} {row['all']:>5} {row['day']:>5} {row['night']:>5}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"model": str(model), "plates": len(plates), "strong_from": strong_from,
                                        "found": results, "per_plate": plates}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
