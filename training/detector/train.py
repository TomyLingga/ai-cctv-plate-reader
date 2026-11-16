"""Train a plate detector with Ultralytics, or measure one, on the dataset of build_dataset.py.

Train (weights are downloaded next to --model when the file is missing):
    uv run --project training python training/detector/train.py train \
        --model models/detector/src/yolo26n.pt --data data/dataset/detector/data.yaml --name yolo26n-20261115

Measure a model on the test split named in a data file:
    uv run --project training python training/detector/train.py val \
        --model data/training/detector/yolo26n-20261115/weights/best.pt --data data/dataset/detector/public.yaml

Runs land in data/training/detector/<name>/. The runtime never loads these .pt files: export the
chosen one with training/detector/export_onnx.py (--nms false for YOLO26, --nms true for YOLOv8).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

RUNS = Path("data/training/detector")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("train", "val"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--name", default=None, help="run name (train) or label of the result file (val)")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=30, help="stop when validation has not improved for this many epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="the runtime feeds 640; keep it unless the runtime changes too")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    # Loader processes each load the CUDA libraries: on the 16 GB laptop two of them next to
    # another training exhausted the page file and the run died ("Couldn't open shared event").
    # Images are kept in RAM instead, which is what makes a single process fast enough.
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache", choices=("ram", "disk", "none"), default="ram")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split", default="test", help="val mode: split of the data file to measure on")
    args = parser.parse_args()

    # Stop Ultralytics from installing packages on its own.
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    data = str(args.data.resolve())
    project = str(RUNS.resolve())
    if args.mode == "train":
        model.train(
            data=data, epochs=args.epochs, patience=args.patience, imgsz=args.imgsz, batch=args.batch,
            device=args.device, workers=args.workers, seed=args.seed, project=project,
            cache=False if args.cache == "none" else args.cache,
            name=args.name or args.model.stem, exist_ok=True,
            # A mirrored plate does not exist; everything else stays at the library's defaults.
            fliplr=0.0,
        )
        return
    metrics = model.val(data=data, split=args.split, imgsz=args.imgsz, batch=args.batch, device=args.device,
                        workers=args.workers, project=project, name=f"val-{args.name or args.model.stem}", exist_ok=True,
                        plots=False, verbose=False)
    result = {
        "model": str(args.model), "data": str(args.data), "split": args.split,
        "precision": round(float(metrics.box.mp), 4), "recall": round(float(metrics.box.mr), 4),
        "map50": round(float(metrics.box.map50), 4), "map50_95": round(float(metrics.box.map), 4),
        "speed_ms": {key: round(float(value), 2) for key, value in metrics.speed.items()},
    }
    print(json.dumps(result))
    out = RUNS / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name or args.model.stem}-{args.data.stem}-{args.split}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
