"""Measure an OCR model on a labelled CSV: plate accuracy and character accuracy.

Runs the ONNX model the same way the service does (alpr_id.ocr.recognizer.PlateOcr), on the
crops exactly as the service saved them. No voting and no correction: this is the model
alone, one crop at a time, so two models can be compared on the same crops.

  plate accuracy      share of crops whose whole text is right
  character accuracy  1 - (edit distance to the label, summed) / (label length, summed)

What the number means depends on the labels. On a CSV of human-checked labels it is
accuracy. On automatic labels it is only how often the model agrees with the plate its own
votes settled on; the script prints which of the two it was given.

Usage (from the project root):
    uv run --frozen --extra cpu python training/ocr/evaluate.py --csv data/dataset/ocr/test.csv
    uv run --frozen --extra cpu python training/ocr/evaluate.py --csv data/dataset/ocr/val.csv \
        --model models/ocr/finetuned.onnx --plate-config models/ocr/cct_s_v2_global_plate_config.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2

from alpr_id.config import load_config
from alpr_id.detect.providers import resolve_providers
from alpr_id.logging_setup import configure_logging
from alpr_id.ocr.recognizer import PlateOcr


def align(label: str, read: str) -> tuple[int, list[tuple[str, str]]]:
    """Edit distance and the substitutions (label char, read char) on one cheapest path."""
    rows, cols = len(label) + 1, len(read) + 1
    cost = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        cost[i][0] = i
    for j in range(cols):
        cost[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost[i][j] = min(cost[i - 1][j] + 1, cost[i][j - 1] + 1, cost[i - 1][j - 1] + (label[i - 1] != read[j - 1]))
    swaps, i, j = [], len(label), len(read)
    while i > 0 and j > 0:
        if cost[i][j] == cost[i - 1][j - 1] + (label[i - 1] != read[j - 1]):
            if label[i - 1] != read[j - 1]:
                swaps.append((label[i - 1], read[j - 1]))
            i, j = i - 1, j - 1
        elif cost[i][j] == cost[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return cost[-1][-1], swaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, required=True, help="image_path,plate_text; paths relative to the CSV or absolute")
    parser.add_argument("--model", type=Path, default=None, help="ONNX model (default: the one in config/app.yaml)")
    parser.add_argument("--plate-config", type=Path, default=None)
    parser.add_argument("--all-csv", type=Path, default=None, help="all.csv of the dataset, to say where the labels came from")
    parser.add_argument("--out", type=Path, default=None, help="write the result as JSON here")
    parser.add_argument("--show-errors", type=int, default=15)
    args = parser.parse_args()

    configure_logging(log_dir=None, level="ERROR")
    config = load_config()
    model = args.model or config.ocr.model_path
    ocr = PlateOcr(model, args.plate_config or config.ocr.plate_config_path, resolve_providers(config.runtime))

    with args.csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{args.csv} has no rows")
    sources: Counter[str] = Counter()
    info: dict[str, dict] = {}
    all_csv = args.all_csv or args.csv.parent / "all.csv"
    if all_csv.is_file():
        with all_csv.open(encoding="utf-8", newline="") as handle:
            info = {row["image_path"]: row for row in csv.DictReader(handle)}

    right = distance_sum = length_sum = 0
    swaps: Counter[tuple[str, str]] = Counter()
    per_camera: dict[str, list[int]] = {}
    errors = []
    reads = []  # one entry per crop, so two models can be compared crop by crop
    for row in rows:
        # The trainer's CSVs hold paths relative to the folder of the CSV.
        path = Path(row["image_path"])
        row["image_path"] = str(path if path.is_absolute() else (args.csv.parent / path).resolve())
        image = cv2.imread(row["image_path"])
        if image is None:
            raise SystemExit(f"cannot read {row['image_path']}")
        read = ocr.read([image])[0]
        label = row["plate_text"]
        distance, pairs = align(label, read.text)
        right += read.text == label
        distance_sum += distance
        length_sum += len(label)
        swaps.update(pairs)
        meta = info.get(row["image_path"], {})
        sources[meta.get("source", "unknown")] += 1
        bucket = per_camera.setdefault(meta.get("camera", "?"), [0, 0])
        bucket[0] += read.text == label
        bucket[1] += 1
        reads.append({"image": row["image_path"], "label": label, "read": read.text,
                      "confidence": round(float(read.confidence), 4), "distance": distance})
        if read.text != label:
            errors.append((label, read.text, round(read.confidence, 2), Path(row["image_path"]).name))

    total = len(rows)
    result = {
        "model": str(model),
        "csv": str(args.csv),
        "crops": total,
        "label_sources": dict(sources),
        "plate_accuracy": round(right / total, 4),
        "character_accuracy": round(1 - distance_sum / max(length_sum, 1), 4),
        "per_camera": {cam: {"right": r, "crops": n} for cam, (r, n) in sorted(per_camera.items())},
        "top_confusions": [{"label": a, "read": b, "count": n} for (a, b), n in swaps.most_common(10)],
    }
    human_only = set(sources) == {"human"}
    print(f"model: {model}")
    print(f"crops: {total} from {args.csv}  (labels: {', '.join(f'{k} {v}' for k, v in sources.items())})")
    print("These labels were checked by a person: the numbers below are accuracy." if human_only else
          "These labels are NOT all human-checked: the numbers below are agreement with the labels, not accuracy.")
    print(f"plate: {right}/{total} = {right / total:.1%}   character: {result['character_accuracy']:.1%}")
    print("per camera: " + ", ".join(f"{cam} {r}/{n}" for cam, (r, n) in sorted(per_camera.items())))
    if swaps:
        print("most frequent substitutions (label -> read): " + ", ".join(f"{a}->{b} x{n}" for (a, b), n in swaps.most_common(10)))
    for label, read, confidence, name in errors[: args.show_errors]:
        print(f"   {label:<10} read as {read or '(empty)':<10} conf {confidence:.2f}  {name}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({**result, "reads": reads}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
