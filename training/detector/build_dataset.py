"""Build the detector dataset: public street photos plus frames of our own cameras.

Sources
  --public NAME=FOLDER   a dataset in YOLO layout (images/<split>/*.jpg, labels/<split>/*.txt) whose
                         plate boxes were drawn by people. Images are scaled down to --public-size.
  --events FOLDER        frames the service saved with each event (<folder>/media/<date>/<camera>/*_frame.jpg).
  --videos FOLDER ...    recordings of the cameras (<folder>/<camera>.mkv); key frames only.

Frames of our own cameras have no boxes drawn by people. They are labelled by the detector and
the OCR that run in the service (the "teacher"), with one safeguard:

  - the frame is cut to the camera's ROI first, exactly as the service does before detection;
  - a proposed box counts as a plate when the OCR reads a standard plate with a known region
    code from it, with confidence of at least --min-read;
  - a frame is used only when every box the detector proposes (down to --propose) is such a
    plate, or when it proposes nothing (background). A frame with a box nobody can vouch for
    is left out, so tail lights and vest lettering the teacher mistakes for plates do not
    become training labels.

What this cannot give: a plate the teacher missed is in no label. A model measured on these
frames can show that it finds what the teacher finds; it cannot show that it finds more.

Splits: public data keeps its own train/val/test. Camera frames from --test-from onwards go to
`test_cctv`, earlier ones to train and val. Two data files are written, `data.yaml` (test =
camera frames) and `public.yaml` (test = the public test split).

Usage (from the project root, runtime environment):
    uv run --frozen --extra cpu python training/detector/build_dataset.py --out data/dataset/detector \
        --public kaggle="../dataset/Indonesian License Plate Dataset" \
        --events data/dataset/raw/rog --videos data/samples/rec-20261113-a data/samples/rec-20261113-b
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import av
import cv2
import numpy as np

from alpr_id.capture.roi import clamp_box, roi_to_pixels
from alpr_id.config import load_config
from alpr_id.detect.onnx_detector import PlateDetector
from alpr_id.detect.providers import create_session, resolve_providers
from alpr_id.logging_setup import configure_logging
from alpr_id.ocr.recognizer import PlateOcr
from alpr_id.postprocess.correction import normalize_plate

SPLITS = ("train", "val", "test_cctv", "test_public")


def bucket(name: str) -> float:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


class Teacher:
    """The service's own detector and OCR, used to label a frame or to refuse it."""

    def __init__(self, propose: float, min_read: float) -> None:
        config = load_config()
        self.config = config
        self.detector = PlateDetector(create_session(config.detector.model_path, config.runtime),
                                      conf_threshold=propose, iou_threshold=config.detector.iou_threshold)
        self.ocr = PlateOcr(config.ocr.model_path, config.ocr.plate_config_path, resolve_providers(config.runtime))
        self.min_read = min_read
        self.roi = {camera.id: tuple(camera.roi) for camera in config.cameras}

    def label(self, frame: np.ndarray, camera: str) -> tuple[np.ndarray, list[tuple[float, float, float, float]], list[str]] | None:
        """ROI crop, plate boxes in that crop, their texts. None when a proposed box can't be vouched for."""
        rect = roi_to_pixels(self.roi[camera], frame.shape[1], frame.shape[0])
        view = rect.crop(frame)
        boxes, texts = [], []
        for found in self.detector.detect(view):
            area = clamp_box(found.box, view.shape[1], view.shape[0], self.config.ocr.crop_padding)
            if area is None:
                return None
            read = self.ocr.read([area.crop(view)])[0]
            plate = normalize_plate(read.text)
            if not (plate.format_valid and plate.region_known and read.confidence >= self.min_read):
                return None
            boxes.append(found.box)
            texts.append(plate.plate)
        return view, boxes, texts


class Writer:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.rows: list[dict] = []
        for split in SPLITS:
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    def add(self, split: str, name: str, image: np.ndarray, boxes: list[tuple[float, float, float, float]],
            source: str, texts: list[str]) -> None:
        height, width = image.shape[:2]
        cv2.imwrite(str(self.out / "images" / split / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        lines = [f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
                 for x1, y1, x2, y2 in boxes]
        (self.out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self.rows.append({"image": f"images/{split}/{name}.jpg", "split": split, "source": source, "boxes": len(boxes),
                          "texts": " ".join(texts), "width": width, "height": height})

    def finish(self) -> None:
        with (self.out / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        root = self.out.resolve().as_posix()
        for name, test in (("data", "test_cctv"), ("public", "test_public")):
            (self.out / f"{name}.yaml").write_text(
                f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/{test}\nnames:\n  0: plat\n", encoding="utf-8")


def add_public(writer: Writer, name: str, folder: Path, size: int) -> None:
    for theirs, ours in (("train", "train"), ("val", "val"), ("valid", "val"), ("test", "test_public")):
        for path in sorted((folder / "images" / theirs).glob("*.jpg")):
            label = folder / "labels" / theirs / (path.stem + ".txt")
            image = cv2.imread(str(path))
            if image is None or not label.is_file():
                continue
            scale = min(1.0, size / max(image.shape[:2]))
            if scale < 1.0:
                image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            height, width = image.shape[:2]
            boxes = []
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 5:  # anything else is a polygon or a broken line
                    x, y, w, h = (float(v) for v in parts[1:])
                    boxes.append(((x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height))
            writer.add(ours, f"pub_{name}_{path.stem}", image, boxes, f"public:{name}", [])


def add_events(writer: Writer, teacher: Teacher, raw: Path, test_from: datetime, val_share: float, counts: Counter) -> None:
    for path in sorted((raw / "media").glob("*/*/*_frame.jpg")):
        date, camera = path.parts[-3], path.parts[-2]
        clock = path.name[:6]
        moment = datetime.fromisoformat(f"{date}T{clock[:2]}:{clock[2:4]}:{clock[4:]}")
        frame = cv2.imread(str(path))
        labelled = teacher.label(frame, camera) if frame is not None and camera in teacher.roi else None
        if labelled is None or not labelled[1]:
            counts["event frames left out"] += 1
            continue
        view, boxes, texts = labelled
        name = f"cctv_ev_{date}_{camera}_{path.stem.removesuffix('_frame')}"
        split = "test_cctv" if moment >= test_from else ("val" if bucket(name) < val_share else "train")
        writer.add(split, name, view, boxes, "cctv:event", texts)
        counts[f"event frames -> {split}"] += 1


def add_videos(writer: Writer, teacher: Teacher, folder: Path, every: float, per_plate: int, backgrounds: int,
               val_share: float, counts: Counter) -> None:
    for path in sorted(folder.glob("*.mkv")):
        camera = path.stem
        if camera not in teacher.roi:
            continue
        kept: Counter[str] = Counter()
        last_seen: dict[str, float] = {}
        last_time, last_background, empty = -1e9, -1e9, 0
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "NONKEY"  # key frames are sharp and cost no extra decoding
            for frame in container.decode(stream):
                moment = float(frame.pts * stream.time_base) if frame.pts is not None else last_time + every
                if moment - last_time < every:
                    continue
                last_time = moment
                counts["video frames looked at"] += 1
                labelled = teacher.label(frame.to_ndarray(format="bgr24"), camera)
                if labelled is None:
                    counts["video frames left out"] += 1
                    continue
                view, boxes, texts = labelled
                name = f"cctv_rec_{folder.name}_{camera}_{int(moment * 10):06d}"
                split = "val" if bucket(name) < val_share else "train"
                if not boxes:
                    if empty < backgrounds and moment - last_background >= 20:
                        writer.add(split, name, view, [], "cctv:video background", [])
                        empty, last_background = empty + 1, moment
                        counts["video backgrounds kept"] += 1
                    continue
                key = "+".join(sorted(texts))
                # A standing truck gives the same picture for minutes: a few of them are enough.
                if kept[key] >= per_plate or moment - last_seen.get(key, -1e9) < 3:
                    continue
                kept[key] += 1
                last_seen[key] = moment
                writer.add(split, name, view, boxes, "cctv:video", texts)
                counts["video frames kept"] += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--public", action="append", default=[], metavar="NAME=FOLDER")
    parser.add_argument("--public-size", type=int, default=1280, help="longest side of public images after scaling down")
    parser.add_argument("--events", type=Path, default=None)
    parser.add_argument("--videos", type=Path, nargs="*", default=[])
    parser.add_argument("--test-from", type=datetime.fromisoformat, default=datetime(2026, 11, 14, 12, 0),
                        help="camera frames from this moment on are the test set")
    parser.add_argument("--propose", type=float, default=0.15, help="detector confidence from which a box has to be explained")
    parser.add_argument("--min-read", type=float, default=0.8, help="OCR confidence needed to call a box a plate")
    parser.add_argument("--every", type=float, default=1.0, help="seconds between video frames looked at")
    parser.add_argument("--per-plate", type=int, default=12, help="frames kept per plate and recording")
    parser.add_argument("--backgrounds", type=int, default=40, help="empty frames kept per recording")
    parser.add_argument("--val-share", type=float, default=0.12)
    args = parser.parse_args()

    configure_logging(log_dir=None, level="ERROR")
    if args.out.exists() and any(args.out.iterdir()):
        if not (args.out / "manifest.csv").is_file():
            raise SystemExit(f"{args.out} exists and was not written by this script; choose another --out")
        shutil.rmtree(args.out)  # rebuilt from its sources every time
    writer = Writer(args.out)
    counts: Counter[str] = Counter()
    for item in args.public:
        name, _, folder = item.partition("=")
        add_public(writer, name, Path(folder), args.public_size)
    if args.events or args.videos:
        teacher = Teacher(args.propose, args.min_read)
        print(f"teacher: detector on {teacher.detector.providers[0]}, OCR on {teacher.ocr.providers[0]}")
        if args.events:
            add_events(writer, teacher, args.events, args.test_from, args.val_share, counts)
        for folder in args.videos:
            add_videos(writer, teacher, folder, args.every, args.per_plate, args.backgrounds, args.val_share, counts)
    writer.finish()

    for key, value in sorted(counts.items()):
        print(f"{value:6d}  {key}")
    for split in SPLITS:
        part = [r for r in writer.rows if r["split"] == split]
        by_source = Counter(r["source"] for r in part)
        print(f"{split:<12} {len(part):5d} images, {sum(r['boxes'] for r in part):5d} boxes  "
              + ", ".join(f"{source} {n}" for source, n in sorted(by_source.items())))
    print(f"written: {args.out / 'data.yaml'}, {args.out / 'public.yaml'}, {args.out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
