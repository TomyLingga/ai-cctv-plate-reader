"""Turn the Kaggle "Indonesian License Plate Recognition Dataset" into a CSV the OCR trainer reads.

The dataset holds one crop per plate (602x189) and a YOLO label file with one box per character.
The plate text is those characters read left to right. Texts outside the plate pattern
(1-2 letters, 1-4 digits, 0-3 letters) are left out: they are labelling slips or special plates.

The crops stay where they were downloaded; the CSV points at them with absolute paths, so it
only works on the machine that wrote it. Nothing in the dataset folder is changed.

Usage (from the project root):
    uv run --frozen --extra cpu python training/dataset/kaggle_recognition_csv.py \
        --dataset "..\\dataset\\Indonesian License Plate Recognition Dataset" \
        --out data/dataset/ocr/extra/kaggle_recognition_clean.csv

Use the result as --extra-csv of training/ocr/crossval.py. These are sharp phone photos: train on
them together with training/ocr/augmentation-cctv.yaml, and never count accuracy on them.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

PLATE = re.compile(r"^[A-Z]{1,2}\d{1,4}[A-Z]{0,3}$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True, help="folder with classes.names, images/ and labels/")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    names = (args.dataset / "classes.names").read_text(encoding="utf-8").split()
    rows, skipped = [], 0
    for image in sorted((args.dataset / "images").rglob("*.jpg")):
        label = args.dataset / "labels" / image.parent.name / (image.stem + ".txt")
        boxes = [line.split() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        # class index, then the box centre: sort by the centre's x to read left to right
        text = "".join(names[int(box[0])] for box in sorted(boxes, key=lambda box: float(box[1])))
        if PLATE.match(text):
            rows.append([str(image.resolve()), text, "Indonesia"])
        else:
            skipped += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "plate_text", "plate_region"])
        writer.writerows(rows)
    print(f"{args.out}: {len(rows)} crops, {skipped} left out (text outside the plate pattern)")


if __name__ == "__main__":
    main()
