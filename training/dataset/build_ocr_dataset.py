"""Turn the crops the service saved into an OCR dataset with fixed splits.

Input is a copy of the service's `data/media` folder and its event list:

    <raw>/media/<date>/<camera>/reads/<HHMMSS>_t<track>_<TEXT>_<conf>_<id>.jpg
    <raw>/events.json                      (GET /events?limit=1000)

Every crop that went through OCR is there, with what the model read from it in the file
name. One vehicle standing in front of a camera gives several crops of the same plate.

Labels come from two places:

  human   `labels.csv` written by the review page (make_review_page.py). Always wins.
  auto    the plate of the vehicle's event, when the event is trustworthy: standard pattern,
          listed region code that was not replaced, confidence of at least --min-confidence,
          and at least two crops of the vehicle that read exactly that plate on their own.
          Crops of the same vehicle that read something else get the same label; those are
          the hard examples the model has to learn.

Auto labels are the model agreeing with itself. They are good enough to train on, not to
measure accuracy with: only crops with a human label count as a test set.

Splits are made per plate, not per crop, so the same truck never sits in train and test:
the split of a plate is decided by a hash of its text and never changes between runs.

Usage:
    uv run --frozen --extra cpu python training/dataset/build_ocr_dataset.py \
        --raw data/dataset/raw/rog --out data/dataset/ocr
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2

READ_NAME = re.compile(r"^(\d{6})_t(\d+)_([A-Z0-9]+)_(\d{3})_[0-9a-f]+$")
SITTING_GAP_S = 15 * 60  # crops of one track id further apart than this are another vehicle
EVENT_MATCH_S = 120
# A whole Indonesian plate is about 3.2 times as wide as it is high. A crop far below that is a
# plate that is partly hidden or cut off.
MIN_ASPECT, MAX_ASPECT = 2.2, 5.5
# The pretrained model has a second output that names the country. The trainer only builds
# that output when the CSV has this column, and the fine-tuned model has to keep the shape
# of the checkpoint it starts from.
PLATE_REGION = "Indonesia"
# Some reviewers type this into the text box instead of picking "Bukan plat" from the list.
NOT_A_PLATE = "NOTPLATE"


@dataclass
class Crop:
    image_path: str
    plate_text: str  # the label, empty when there is none
    source: str  # human | auto | none
    split: str  # train | val | test | none
    camera: str
    date: str
    time: str
    track: int
    own_text: str  # what the OCR read from this crop
    own_confidence: float
    agrees: bool  # own_text equals the label
    suspect: str  # why this crop is left out of training by default, or empty
    width: int
    height: int


def distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def split_of(plate: str, val_share: float, test_share: float) -> str:
    """Stable split for a plate: the same text always lands in the same split."""
    bucket = int(hashlib.sha256(plate.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < test_share:
        return "test"
    return "val" if bucket < test_share + val_share else "train"


def read_crops(media: Path) -> list[dict]:
    found = []
    for path in sorted(media.glob("*/*/reads/*.jpg")):
        match = READ_NAME.match(path.stem)
        if not match:
            continue
        clock, track, text, conf = match.groups()
        date, camera = path.parts[-4], path.parts[-3]
        found.append({
            "path": path,
            "date": date,
            "camera": camera,
            "time": f"{clock[:2]}:{clock[2:4]}:{clock[4:]}",
            "moment": datetime.fromisoformat(f"{date}T{clock[:2]}:{clock[2:4]}:{clock[4:]}"),
            "track": int(track),
            "text": "" if text == "EMPTY" else text,
            "confidence": int(conf) / 100.0,
        })
    return found


def sittings(crops: list[dict]) -> list[list[dict]]:
    """Group crops into vehicles: same camera and track id, without a long gap in between."""
    by_track: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for crop in crops:
        by_track[(crop["date"], crop["camera"], crop["track"])].append(crop)
    groups = []
    for items in by_track.values():
        items.sort(key=lambda c: c["moment"])
        current = [items[0]]
        for crop in items[1:]:
            if (crop["moment"] - current[-1]["moment"]).total_seconds() > SITTING_GAP_S:
                groups.append(current)
                current = []
            current.append(crop)
        groups.append(current)
    return groups


def event_for(group: list[dict], events: list[dict]) -> dict | None:
    camera, track = group[0]["camera"], group[0]["track"]
    start, end = group[0]["moment"], group[-1]["moment"]
    for event in events:
        if event["camera_id"] != camera or event["track_id"] != track:
            continue
        moment = datetime.fromisoformat(event["timestamp"]).replace(tzinfo=None)
        if (start - moment).total_seconds() <= EVENT_MATCH_S and (moment - end).total_seconds() <= EVENT_MATCH_S:
            return event
    return None


def verdict_of(row: dict) -> tuple[str, str]:
    """The reviewer's text and status for one crop, with the typed NOTPLATE turned into a status."""
    text = row["plate"].strip().upper()
    if text == NOT_A_PLATE:
        return "", "not_plate"
    return text, row.get("status") or "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, required=True, help="folder with media/ and events.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None, help="labels.csv from the review page (default <out>/review/labels.csv)")
    parser.add_argument("--min-confidence", type=float, default=0.9, help="event confidence needed for an auto label")
    parser.add_argument("--val-share", type=float, default=0.15)
    parser.add_argument("--test-share", type=float, default=0.15)
    args = parser.parse_args()

    events = json.loads((args.raw / "events.json").read_text(encoding="utf-8"))
    crops = read_crops(args.raw / "media")
    labels_path = args.labels or args.out / "review" / "labels.csv"
    human: dict[str, dict] = {}
    if labels_path.is_file():
        with labels_path.open(encoding="utf-8-sig", newline="") as handle:
            human = {row["image"]: row for row in csv.DictReader(handle)}

    rows: list[Crop] = []
    for group in sittings(crops):
        event = event_for(group, events)
        auto = ""
        if event and event["format_valid"] and event["region_known"] and not event.get("region_corrected") \
                and event["confidence"] >= args.min_confidence \
                and sum(crop["text"] == event["plate"] for crop in group) >= 2:
            auto = event["plate"]
        # What the reviewer wrote for this vehicle, to catch a typing slip on one of its crops.
        # The page pre-fills the box with the OCR read, so a text left untouched on a crop
        # marked unreadable says nothing and does not count.
        names = [crop["path"].relative_to(args.raw / "media").as_posix() for crop in group]
        stated: Counter[str] = Counter()
        for crop, name in zip(group, names, strict=True):
            if name in human:
                text, status = verdict_of(human[name])
                if text and (status in ("ok", "covered") or text != crop["text"]):
                    stated[text] += 1
        for crop, name in zip(group, names, strict=True):
            image = cv2.imread(str(crop["path"]))
            height, width = image.shape[:2] if image is not None else (0, 0)
            label, source, suspect = auto, ("auto" if auto else "none"), ""
            if name in human:
                (label, status), source = verdict_of(human[name]), "human"
                if status == "not_plate":
                    # Kept in all.csv: these are the negatives the detector has to learn.
                    label, suspect = "", "marked as not a plate by the reviewer"
                elif status == "unreadable" or not label:
                    label, suspect = "", "marked as unreadable by the reviewer"
                elif status == "covered":
                    suspect = "marked as covered by the reviewer"
                elif any(text != label and count >= stated[label] for text, count in stated.items()):
                    suspect = "label differs from the other crops of the same vehicle"
            # A person who looked at the crop and called it whole outranks the guesses below.
            if label and not suspect and source == "auto":
                aspect = width / height if height else 0.0
                if not MIN_ASPECT <= aspect <= MAX_ASPECT:
                    suspect = f"crop shape {aspect:.1f}:1, a whole plate is about 3.2:1"
                elif source == "auto" and distance(crop["text"], label) > 3:
                    suspect = "read too different from the label; may be another vehicle"
                elif len(crop["text"]) <= len(label) - 2:
                    suspect = "read is two or more characters short: part of the plate may be hidden"
            rows.append(Crop(
                image_path=str(crop["path"].resolve()),
                plate_text=label,
                source=source,
                split=split_of(label, args.val_share, args.test_share) if label else "none",
                camera=crop["camera"], date=crop["date"], time=crop["time"], track=crop["track"],
                own_text=crop["text"], own_confidence=crop["confidence"],
                agrees=bool(label) and crop["text"] == label,
                suspect=suspect, width=width, height=height,
            ))

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "all.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    # The files the trainer reads: clean crops only. The test file holds human labels only.
    # fast-plate-ocr joins the folder of the CSV and image_path as text, so the paths have
    # to be relative to that folder.
    here = str(args.out.resolve())
    for split in ("train", "val", "test"):
        chosen = [r for r in rows if r.split == split and r.plate_text and not r.suspect and (split != "test" or r.source == "human")]
        with (args.out / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path", "plate_text", "plate_region"])
            writer.writerows([os.path.relpath(r.image_path, here), r.plate_text, PLATE_REGION] for r in chosen)
    # For the question "do covered plates hurt?": the same training crops plus the covered
    # ones, and the covered crops of the test plates on their own.
    covered = [r for r in rows if r.plate_text and r.suspect.startswith("marked as covered")]
    extras = {
        "train_with_covered": [r for r in rows if r.split == "train" and r.plate_text and (not r.suspect or r in covered)],
        "test_covered": [r for r in covered if r.split == "test"],
    }
    for name, chosen in extras.items():
        with (args.out / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_path", "plate_text", "plate_region"])
            writer.writerows([os.path.relpath(r.image_path, here), r.plate_text, PLATE_REGION] for r in chosen)

    labelled = [r for r in rows if r.plate_text]
    clean = [r for r in labelled if not r.suspect]
    print(f"crops: {len(rows)}  |  labelled: {len(labelled)} (human {sum(r.source == 'human' for r in labelled)}, "
          f"auto {sum(r.source == 'auto' for r in labelled)})  |  without a label: {len(rows) - len(labelled)}")
    print(f"labelled and clean: {len(clean)}  |  left out as suspect: {len(labelled) - len(clean)}")
    for reason, count in Counter(r.suspect.split(":")[0].split(",")[0] for r in labelled if r.suspect).most_common():
        print(f"    {count:4d}  {reason}")
    for reason, count in Counter(r.suspect for r in rows if r.suspect and not r.plate_text).most_common():
        print(f"    {count:4d}  {reason} (no label kept)")
    print(f"covered crops: {len(covered)}, of which train {sum(r.split == 'train' for r in covered)}, "
          f"val {sum(r.split == 'val' for r in covered)}, test {sum(r.split == 'test' for r in covered)}")
    print(f"distinct plates among clean crops: {len({r.plate_text for r in clean})}")
    wrong = [r for r in clean if not r.agrees]
    print(f"clean crops whose own read differs from the label (hard examples): {len(wrong)} of {len(clean)}")
    for split in ("train", "val", "test"):
        part = [r for r in clean if r.split == split]
        print(f"  {split:<5} {len(part):4d} crops, {len({r.plate_text for r in part}):3d} plates, "
              f"{sum(r.source == 'human' for r in part)} with a human label")
    print("per camera (clean): " + ", ".join(f"{cam} {n}" for cam, n in sorted(Counter(r.camera for r in clean).items())))
    print(f"written: {args.out / 'all.csv'} and train/val/test.csv")


if __name__ == "__main__":
    main()
