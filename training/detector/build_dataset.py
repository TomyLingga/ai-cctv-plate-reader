"""Build the detector dataset: public street photos plus frames of our own cameras.

Sources
  --public NAME=FOLDER   a dataset in YOLO layout (images/<split>/*.jpg, labels/<split>/*.txt) whose
                         plate boxes were drawn by people. Images are scaled down to --public-size.
  --events FOLDER        frames the service saved with each event (<folder>/media/<date>/<camera>/*_frame.jpg)
                         and, when it is there, the event database copied with them (<folder>/alpr.sqlite3).
  --videos FOLDER ...    recordings of the cameras (<folder>/<camera>.mkv); key frames only.

Frames of our own cameras have no boxes drawn by people. They are labelled by the detector and
the OCR that run in the service (the "teacher"):

  - The detector looks at the frame the way the service does (the camera's ROI shrunk to the
    detector input, and the camera's focus area) and also at every part of the ROI at full
    resolution, tile by tile. A plate too small for the shrunk ROI is found in a tile. That is
    how plates the service misses get a label: standing plates about 105 pixels wide, and
    plates at night (found on the weighbridge on 2026-11-17).
  - A second detector (--second-detector, trained by other people on other pictures) looks at
    the focus area and the tiles too. Where both see a plate the box of the service's detector
    is kept; the second one adds the plates the first is blind to.
  - A proposed box (down to --propose) counts as a plate when the OCR reads a standard plate
    of at least six characters with a known region code from it with confidence of at least
    --min-read, or when the service reported such a plate with that confidence at that place
    in that frame. Shorter texts are the visible half of a covered plate. An event
    alone proves nothing: the service has reported a kerb stone (E6G) and vest lettering
    (SECUR1TY) as plates.
  - A box nobody can vouch for is neither a plate nor background. It is painted over in the
    training pictures: tail lights, vest lettering, plates too dirty to read.

Every frame gives up to two training pictures, the two views the service's detector gets: the
ROI and the focus area. A plate cut by the edge of a picture is painted over in that picture.

What this cannot give: a plate both detectors miss in every view is in no label. A model
measured on these frames can show that it finds, in the views the service uses, what the
teacher finds with all its tiles; it cannot show that it finds more than that.

Background pictures (--backgrounds, video frames in which nothing was proposed) are off by
default. "The teacher sees nothing" is no proof that a frame is empty: the dataset of
2026-11-18 took 486 such pictures, dozens of them tank trucks from behind whose plate the
service's detector cannot see, and the model trained on them gave no box at all for that kind
of plate. Turn them on only for recordings looked through by a person.

Splits: public data keeps its own train/val/test. Camera frames from --test-from onwards go to
`test_cctv`, earlier ones to train and val; both pictures of a frame go to the same split. Two
data files are written, `data.yaml` (test = camera frames) and `public.yaml` (test = the
public test split). `manifest.csv` says for every picture which view it is and whether it was
taken at night.

Usage (from the project root, runtime environment):
    uv run --frozen --extra cpu python training/detector/build_dataset.py --out data/dataset/detector-20261118 \
        --public kaggle="../dataset/Indonesian License Plate Dataset" --test-from 2026-11-18T00:00 \
        --events data/rog-20261118 --videos data/samples/rec-20261113-a data/samples/rec-20261113-b
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import av
import cv2
import numpy as np

from alpr_id.capture.roi import PixelRect, clamp_box, roi_to_pixels
from alpr_id.config import load_config
from alpr_id.detect.onnx_detector import PlateDetector
from alpr_id.detect.providers import create_session
from alpr_id.logging_setup import configure_logging
from alpr_id.ocr.recognizer import PlateOcr
from alpr_id.pipeline.camera_pipeline import merge_detections
from alpr_id.postprocess.correction import normalize_plate
from alpr_id.track.bytetrack import box_iou

SPLITS = ("train", "val", "test_cctv", "test_public")
Box = tuple[float, float, float, float]

PAD_VALUE = 114  # the grey the detector's letterbox pads with: "nothing to see here"
EDGE_PX = 3  # a box this close to the edge of a view is a plate cut in half
TILE = 640  # the detector's input size: a tile this big is searched without shrinking
TILE_OVERLAP = 160  # wider than a standing plate, so each one lies whole in some tile
VOUCH_IOU = 0.3  # overlap with the box of an event that makes a proposal that event's plate
SAME_PLATE_IOU = 0.3  # overlap from which a box of the second detector is a plate the first one found too
# Fewer characters than this and the text is the visible half of a covered plate (D85HF behind a pipe,
# BM955 behind a wheel). Covered plates are not training data: a detector taught to box them makes the
# service report half a plate.
MIN_PLATE_CHARS = 6
WHOLE_SHARE = 0.95  # share of a plate box that must lie inside a picture to be labelled there
NIGHT_FROM, NIGHT_UNTIL = 18, 6  # hours of the server clock


def bucket(name: str) -> float:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def tiles(area: PixelRect) -> list[PixelRect]:
    """Squares of TILE pixels that cover `area`, neighbours sharing at least TILE_OVERLAP pixels."""

    def starts(low: int, high: int) -> list[int]:
        last = max(high - TILE, low)
        found = list(range(low, last, TILE - TILE_OVERLAP))
        return [*found, last]

    return [
        PixelRect(x, y, min(x + TILE, area.x2), min(y + TILE, area.y2))
        for y in starts(area.y1, area.y2)
        for x in starts(area.x1, area.x2)
    ]


@dataclass
class Labelled:
    """What the teacher says about one frame, in frame pixels."""

    plates: list[tuple[Box, str]] = field(default_factory=list)
    doubts: list[Box] = field(default_factory=list)


class Teacher:
    """The service's own detector and OCR, and a second detector, looking at a frame from every view."""

    def __init__(self, propose: float, min_read: float, second: Path | None = None) -> None:
        config = load_config()
        self.config = config
        self.detector = PlateDetector(create_session(config.detector.model_path, config.runtime),
                                      conf_threshold=propose, iou_threshold=config.detector.iou_threshold)
        # Not a better detector, a different one: what it sees and the service's detector does not
        # is exactly what a fine-tune must not learn to call background.
        self.second = None if second is None else PlateDetector(
            create_session(second, config.runtime), conf_threshold=propose, iou_threshold=config.detector.iou_threshold)
        # The OCR always runs on the CPU here. On the laptop's RTX 5050 the CUDA provider reads
        # rubbish from every crop (0 of 400 the same as on the CPU, measured 2026-11-18) while both
        # detectors answer the same on GPU and CPU. A teacher that cannot read vouches for nothing.
        self.ocr = PlateOcr(config.ocr.model_path, config.ocr.plate_config_path, ["CPUExecutionProvider"])
        self.min_read = min_read
        self.roi = {camera.id: tuple(camera.roi) for camera in config.cameras}
        self.focus = {camera.id: camera.focus for camera in config.cameras}

    def views(self, shape: tuple[int, ...], camera: str) -> tuple[PixelRect, PixelRect | None]:
        """The two pictures the service's detector gets from a frame of this size: ROI and focus area."""
        height, width = shape[:2]
        focus = self.focus[camera]
        return (roi_to_pixels(self.roi[camera], width, height),
                roi_to_pixels(focus, width, height) if focus is not None else None)

    def label(self, frame: np.ndarray, camera: str, known: list[tuple[Box, str]] | None = None) -> Labelled:
        """Plates and doubtful boxes of a frame. `known` are boxes of events the service reported in it."""
        roi, focus = self.views(frame.shape, camera)

        def whole(d, view: PixelRect) -> bool:
            return d.x1 > EDGE_PX and d.y1 > EDGE_PX and d.x2 < view.width - EDGE_PX and d.y2 < view.height - EDGE_PX

        labelled = Labelled()
        found = []
        for detection in self.detector.detect(roi.crop(frame)):
            if whole(detection, roi):
                found.append(detection.shifted(roi.x1, roi.y1))
            else:
                # Cut by the edge of the ROI: the OCR reads such a crop confidently and wrongly.
                labelled.doubts.append(detection.shifted(roi.x1, roi.y1).box)
        others = []
        for view in [*([focus] if focus is not None else []), *tiles(roi)]:
            picture = view.crop(frame)
            inner = [d.shifted(view.x1, view.y1) for d in self.detector.detect(picture) if whole(d, view)]
            found = merge_detections(found, inner, self.config.detector.iou_threshold)
            if self.second is not None:
                inner = [d.shifted(view.x1, view.y1) for d in self.second.detect(picture) if whole(d, view)]
                others = merge_detections(others, inner, self.config.detector.iou_threshold)
        # The two detectors draw a plate slightly differently. The service's box wins, so the
        # fine-tune keeps drawing boxes the way the OCR crop expects them.
        found += [d for d in others if all(box_iou(d.box, known_box.box) < SAME_PLATE_IOU for known_box in found)]

        for detection in found:
            text = self._vouch(frame, detection.box, known or [])
            if text is None:
                labelled.doubts.append(detection.box)
            else:
                labelled.plates.append((detection.box, text))
        return labelled

    def _vouch(self, frame: np.ndarray, box: Box, known: list[tuple[Box, str]]) -> str | None:
        """The plate text if the OCR or an event of the service stands behind this box."""
        area = clamp_box(box, frame.shape[1], frame.shape[0], self.config.ocr.crop_padding)
        if area is not None:
            read = self.ocr.read([area.crop(frame)])[0]
            plate = normalize_plate(read.text)
            if (plate.format_valid and plate.region_known and read.confidence >= self.min_read
                    and len(plate.plate) >= MIN_PLATE_CHARS):
                return plate.plate
        for event_box, text in known:
            if box_iou(box, event_box) >= VOUCH_IOU and len(text) >= MIN_PLATE_CHARS:
                return text
        return None


def cut(frame: np.ndarray, view: PixelRect, labelled: Labelled) -> tuple[np.ndarray, list[Box], list[str]]:
    """The picture of `view` with the plates that lie whole in it; everything unsure is painted over."""
    picture = view.crop(frame).copy()

    def inside(box: Box) -> tuple[int, int, int, int] | None:
        x1, y1 = max(int(round(box[0])) - view.x1, 0), max(int(round(box[1])) - view.y1, 0)
        x2, y2 = min(int(round(box[2])) - view.x1, view.width), min(int(round(box[3])) - view.y1, view.height)
        return (x1, y1, x2, y2) if x2 - x1 >= 2 and y2 - y1 >= 2 else None

    boxes, texts, keep = [], [], []
    for box in labelled.doubts:
        part = inside(box)
        if part is not None:
            picture[part[1] : part[3], part[0] : part[2]] = PAD_VALUE
    for box, text in labelled.plates:
        part = inside(box)
        if part is None:
            continue
        full = (box[2] - box[0]) * (box[3] - box[1])
        if (part[2] - part[0]) * (part[3] - part[1]) >= WHOLE_SHARE * full:
            boxes.append((box[0] - view.x1, box[1] - view.y1, box[2] - view.x1, box[3] - view.y1))
            texts.append(text)
            keep.append(part)
        else:
            picture[part[1] : part[3], part[0] : part[2]] = PAD_VALUE  # a plate cut by the edge of this picture
    original = view.crop(frame)
    for x1, y1, x2, y2 in keep:  # a doubtful box next to a plate must not eat into the plate
        picture[y1:y2, x1:x2] = original[y1:y2, x1:x2]
    boxes = [(max(b[0], 0.0), max(b[1], 0.0), min(b[2], float(view.width)), min(b[3], float(view.height))) for b in boxes]
    return picture, boxes, texts


class Writer:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.rows: list[dict] = []
        for split in SPLITS:
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    def add(self, split: str, name: str, image: np.ndarray, boxes: list[Box], source: str, texts: list[str],
            view: str = "", night: bool | None = None) -> None:
        height, width = image.shape[:2]
        cv2.imwrite(str(self.out / "images" / split / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        lines = [f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
                 for x1, y1, x2, y2 in boxes]
        (self.out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self.rows.append({"image": f"images/{split}/{name}.jpg", "split": split, "source": source, "boxes": len(boxes),
                          "texts": " ".join(texts), "width": width, "height": height, "view": view,
                          "night": "" if night is None else int(night)})

    def add_views(self, teacher: Teacher, split: str, name: str, frame: np.ndarray, camera: str, labelled: Labelled,
                  source: str, night: bool | None, counts: Counter) -> None:
        """Write the ROI picture and the focus picture of a frame. A focus picture without a plate is skipped
        for frames that have one elsewhere: it would only say "background" about a truck."""
        roi, focus = teacher.views(frame.shape, camera)
        for view_name, view in (("wide", roi), ("focus", focus)):
            if view is None:
                continue
            picture, boxes, texts = cut(frame, view, labelled)
            if not boxes and labelled.plates:
                counts[f"{view_name} pictures without a whole plate"] += 1
                continue
            self.add(split, f"{name}_{view_name[0]}", picture, boxes, source, texts, view=view_name, night=night)
            counts[f"{source} {view_name} -> {split}"] += 1

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


def event_boxes(raw: Path, min_confidence: float) -> dict[str, list[tuple[Box, str]]]:
    """Frame path below media/ -> boxes and plates of the events that can vouch for a box in that frame:
    a standard plate with a known region code, voted with at least `min_confidence`."""
    database = raw / "alpr.sqlite3"
    if not database.is_file():
        return {}
    known: dict[str, list[tuple[Box, str]]] = {}
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        for url, bbox, plate in connection.execute(
                "SELECT snapshot_url, bbox, plate FROM events WHERE snapshot_url IS NOT NULL AND bbox IS NOT NULL"
                " AND format_valid = 1 AND region_known = 1 AND confidence >= ?", (min_confidence,)):
            box = json.loads(bbox)
            if len(box) == 4:
                known.setdefault(url.removeprefix("/media/"), []).append((tuple(float(v) for v in box), plate))
    finally:
        connection.close()
    return known


def is_night(hour: int) -> bool:
    return hour >= NIGHT_FROM or hour < NIGHT_UNTIL


def add_events(writer: Writer, teacher: Teacher, raw: Path, test_from: datetime, val_share: float, counts: Counter) -> None:
    known = event_boxes(raw, teacher.min_read)
    counts["events that can vouch for a box"] = sum(len(v) for v in known.values())
    for path in sorted((raw / "media").glob("*/*/*_frame.jpg")):
        date, camera = path.parts[-3], path.parts[-2]
        clock = path.name[:6]
        moment = datetime.fromisoformat(f"{date}T{clock[:2]}:{clock[2:4]}:{clock[4:]}")
        frame = cv2.imread(str(path))
        if frame is None or camera not in teacher.roi:
            counts["event frames unreadable or of an unknown camera"] += 1
            continue
        labelled = teacher.label(frame, camera, known.get(f"{date}/{camera}/{path.name}"))
        if not labelled.plates:
            counts["event frames without a plate anyone vouches for"] += 1
            continue
        name = f"cctv_ev_{date}_{camera}_{path.stem.removesuffix('_frame')}"
        split = "test_cctv" if moment >= test_from else ("val" if bucket(name) < val_share else "train")
        writer.add_views(teacher, split, name, frame, camera, labelled, "cctv:event", is_night(moment.hour), counts)


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
                image = frame.to_ndarray(format="bgr24")
                labelled = teacher.label(image, camera)
                name = f"cctv_rec_{folder.name}_{camera}_{int(moment * 10):06d}"
                split = "val" if bucket(name) < val_share else "train"
                if not labelled.plates:
                    # Background only when nothing at all was proposed: a doubtful box may be a plate.
                    # Even then the frame may show a plate no detector sees (see the module text).
                    if not labelled.doubts and empty < backgrounds and moment - last_background >= 20:
                        writer.add_views(teacher, split, name, image, camera, labelled, "cctv:video background", None, counts)
                        empty, last_background = empty + 1, moment
                    continue
                key = "+".join(sorted(text for _, text in labelled.plates))
                # A standing truck gives the same picture for minutes: a few of them are enough.
                if kept[key] >= per_plate or moment - last_seen.get(key, -1e9) < 3:
                    continue
                kept[key] += 1
                last_seen[key] = moment
                writer.add_views(teacher, split, name, image, camera, labelled, "cctv:video", None, counts)


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
    parser.add_argument("--backgrounds", type=int, default=0,
                        help="video frames without any proposal kept per recording as background; off by default, see above")
    parser.add_argument("--second-detector", type=Path, default=Path("models/detector/yolo-v9-t-640-license-plates-end2end.onnx"),
                        help="another plate detector that proposes boxes next to the service's own")
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
        teacher = Teacher(args.propose, args.min_read, args.second_detector)
        print(f"teacher: detector on {teacher.detector.providers[0]}, second detector {args.second_detector.name}, "
              f"OCR on {teacher.ocr.providers[0]}", flush=True)
        if args.events:
            add_events(writer, teacher, args.events, args.test_from, args.val_share, counts)
            print("event frames done", flush=True)
        for folder in args.videos:
            add_videos(writer, teacher, folder, args.every, args.per_plate, args.backgrounds, args.val_share, counts)
            print(f"{folder} done", flush=True)
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
