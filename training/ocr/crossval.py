"""Cross-validate a fine-tune by plate: does it read plates it never saw better than the baseline model?

The fixed test split of a small dataset holds a handful of plates, too few to trust. Here every
plate lands in one of K folds (by a hash of its text). For each fold a model is fine-tuned on
the crops of all other plates and then reads the human-labelled crops of that fold, so each
human-labelled crop is read once by a model that never saw its plate. The baseline model (the
hub model, --baseline) reads the same crops, and the two are compared crop by crop. In the
result files the baseline is still called "current", from when it was the model in the service.

Nothing is selected with the held-out fold: every run uses the same settings and the weights
of the last epoch. The trainer insists on a validation file; it gets a slice of the training
crops and its numbers are ignored.

Progress is printed one line per epoch and also appended to <out>/progress.log. A fold that
has finished is not trained again, so an interrupted run continues where it stopped.

Runs from the project root in the runtime environment; training and export are started in
the training environment (training/ocr):

    uv run --frozen --extra cpu python training/ocr/crossval.py --out data/training/ocr/cv-<name> \
        --epochs 25 --lr 0.0005 [--with-covered]

With --final there are no folds: the same settings train one model on every usable crop and
write <out>/model.onnx. That is the model to install once a cross-validation liked the settings.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PRETRAINED = Path("models/ocr/pretrained")
PLATE_REGION = "Indonesia"
HERE = Path(__file__).resolve().parent
_print_lock = threading.Lock()


def training_tool(name: str) -> list[str]:
    """A program of the training environment: straight from its venv when that exists."""
    program = HERE / ".venv" / ("Scripts" if os.name == "nt" else "bin") / (name + (".exe" if os.name == "nt" else ""))
    return [str(program)] if program.is_file() else ["uv", "run", "--project", str(HERE), name]


def fold_of(plate: str, folds: int) -> int:
    return int(hashlib.sha256(plate.encode()).hexdigest()[8:16], 16) % folds


def say(out: Path, text: str) -> None:
    line = f"{time.strftime('%H:%M:%S')}  {text}"
    with _print_lock:
        print(line, flush=True)
        with (out / "progress.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "plate_text", "plate_region"])
        writer.writerows([os.path.relpath(r["image_path"], path.parent.resolve()), r["plate_text"], PLATE_REGION] for r in rows)


def run(command: list[str], log: Path, env: dict[str, str], each_tick=None) -> None:
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(command)}\n")
        handle.flush()
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
        while process.poll() is None:
            time.sleep(2)
            if each_tick:
                each_tick()
        if each_tick:
            each_tick()
    if process.returncode != 0:
        raise SystemExit(f"command failed with exit code {process.returncode}; see {log}")


def epoch_reporter(runs: Path, name: str, epochs: int, out: Path):
    """Returns a function that prints the epochs the trainer has logged since the last call."""
    state = {"seen": 0, "started": time.perf_counter()}

    def report() -> None:
        logs = sorted(runs.glob("*/training_log.csv"))
        if not logs:
            return
        try:
            with logs[-1].open(encoding="utf-8", newline="") as handle:
                rows = [row for row in csv.DictReader(handle) if row.get("plate_acc") not in (None, "")]
        except OSError:  # the trainer is writing the file right now
            return
        for row in rows[state["seen"]:]:
            done = int(row["epoch"]) + 1
            pace = (time.perf_counter() - state["started"]) / done
            say(out, f"{name}  epoch {done:2d}/{epochs}  loss {float(row['loss']):.4f}  "
                     f"plates right on training crops {float(row['plate_acc']):.1%}  "
                     f"({pace:.0f} s per epoch, about {pace * (epochs - done) / 60:.0f} min left in this fold)")
        state["seen"] = len(rows)

    return report


def tool_env() -> dict[str, str]:
    return {**os.environ, "KERAS_BACKEND": "tensorflow", "TF_CPP_MIN_LOG_LEVEL": "2"}


def fit(folder: Path, name: str, train: list[dict], args: argparse.Namespace) -> Path:
    """Fine-tune on `train` with the settings of `args` and return the exported ONNX model."""
    write_csv(folder / "train.csv", train)
    write_csv(folder / "val.csv", train[:: max(1, len(train) // 64)])
    # Training may run on another Keras backend (torch has a GPU on Windows, TensorFlow has not).
    # The export always runs on TensorFlow: that path gives the output names the service expects.
    env = tool_env()
    train_env = {**env, "KERAS_BACKEND": args.backend}
    log = folder / "log.txt"
    runs = folder / f"run-{time.strftime('%Y%m%d-%H%M%S')}"
    say(args.out, f"{name}  training on {len(train)} crops of {len({r['plate_text'] for r in train})} plates")
    run([*training_tool("fast-plate-ocr"), "train",
         "--model-config-file", str(PRETRAINED / "cct_s_v2_global_model_config.yaml"),
         "--plate-config-file", str(PRETRAINED / "cct_s_v2_global_plate_config.yaml"),
         "--annotations", str(folder / "train.csv"), "--val-annotations", str(folder / "val.csv"),
         "--weights-path", str(PRETRAINED / "cct_s_v2_global.keras"),
         "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--lr", str(args.lr),
         "--weight-decay", str(args.weight_decay), "--label-smoothing", str(args.label_smoothing),
         "--no-use-ema", "--output-dir", str(runs), "--seed", str(args.seed),
         *(["--augmentation-path", str(args.augmentation)] if args.augmentation else [])],
        log, train_env, epoch_reporter(runs, name, args.epochs, args.out))
    trained = max(p for p in runs.iterdir() if p.is_dir())  # the trainer names its folder by start time
    model = folder / "model.onnx"
    say(args.out, f"{name}  exporting to ONNX")
    run([*training_tool("python"), str(HERE / "export_onnx.py"),
         "--model", str(trained / "last.keras"), "--plate-config", str(trained / "plate_config.yaml"), "--out", str(model),
         *(["--blend-with", str(PRETRAINED / "cct_s_v2_global.keras"), "--blend-share", str(args.blend_share)]
           if args.blend_share < 1 else [])], log, env)
    return model


def one_fold(fold: int, train: list[dict], test: list[dict], args: argparse.Namespace) -> dict:
    folder = args.out / f"fold-{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / "result.json"
    name = f"fold {fold + 1}/{args.folds}"
    if result_path.is_file():  # finished in an earlier, interrupted run
        say(args.out, f"{name}  already done, skipped")
        return json.loads(result_path.read_text(encoding="utf-8"))
    write_csv(folder / "test.csv", test)
    env = tool_env()
    log = folder / "log.txt"
    started = time.perf_counter()
    say(args.out, f"{name}  held out: {len(test)} crops of {len({r['plate_text'] for r in test})} plates")
    model = fit(folder, name, train, args)
    evaluate = [sys.executable, str(HERE / "evaluate.py"), "--csv", str(folder / "test.csv"),
                "--all-csv", str(args.dataset / "all.csv"), "--show-errors", "0"]
    plate_config = ["--plate-config", str(PRETRAINED / "cct_s_v2_global_plate_config.yaml")]
    run([*evaluate, "--model", str(args.baseline), *plate_config, "--out", str(folder / "eval-current.json")], log, env)
    run([*evaluate, "--model", str(model), *plate_config, "--out", str(folder / "eval-finetuned.json")], log, env)
    current = json.loads((folder / "eval-current.json").read_text(encoding="utf-8"))["reads"]
    tuned = json.loads((folder / "eval-finetuned.json").read_text(encoding="utf-8"))["reads"]
    crops = [{"image": a["image"], "label": a["label"], "current": a["read"], "finetuned": b["read"],
              "current_distance": a["distance"], "finetuned_distance": b["distance"]}
             for a, b in zip(current, tuned, strict=True)]
    result = {"fold": fold, "train_crops": len(train), "train_plates": len({r["plate_text"] for r in train}),
              "test_plates": len({r["plate_text"] for r in test}), "seconds": round(time.perf_counter() - started),
              "crops": crops}
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    say(args.out, f"{name}  done in {result['seconds'] / 60:.1f} min: baseline model "
                  f"{sum(c['current'] == c['label'] for c in crops)}/{len(crops)} right, fine-tuned "
                  f"{sum(c['finetuned'] == c['label'] for c in crops)}/{len(crops)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset/ocr"), help="folder with all.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--only-fold", type=int, default=None, help="run this fold alone (0-based), for a quick check")
    parser.add_argument("--jobs", type=int, default=1, help="folds trained at the same time")
    parser.add_argument("--with-covered", action="store_true", help="also train on crops the reviewer marked as covered")
    parser.add_argument("--extra-csv", type=Path, action="append", default=[],
                        help="crops from another source (image_path, plate_text) added to the training crops of every fold, "
                             "to see whether they help or hurt on our own crops; may be given more than once")
    parser.add_argument("--extra-limit", type=int, default=None, help="use this many of the extra crops (a fixed selection)")
    parser.add_argument("--augmentation", type=Path, default=None,
                        help="augmentation YAML for the trainer (default: the trainer's own); see make_augmentation.py")
    parser.add_argument("--backend", choices=("tensorflow", "torch", "jax"), default="tensorflow",
                        help="Keras backend for training (torch and jax must be installed in training/ocr first; torch on the laptop GPU was not faster than TensorFlow on its CPU); the export always uses tensorflow")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--baseline", type=Path, default=Path("models/ocr/cct_s_v2_global.onnx"),
                        help="model the fold models are compared with. Keep it the hub model: a model made with --final "
                             "has seen every plate, so as a baseline it would look better than it is")
    parser.add_argument("--blend-share", type=float, default=1.0,
                        help="below 1: the exported model averages its weights with the pretrained ones; "
                             "this is the share of the fine-tuned weights (see export_onnx.py --blend-with)")
    parser.add_argument("--final", action="store_true",
                        help="no folds: fine-tune once on every usable crop plus the extra crops and write <out>/model.onnx; "
                             "run it with the settings a cross-validation was happy with")
    parser.add_argument("--final-split", action="append", default=[],
                        help="with --final: only the crops of this split of all.csv (train, val, test); may be given more than once")
    args = parser.parse_args()

    with (args.dataset / "all.csv").open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["plate_text"]]
    usable = [r for r in rows if not r["suspect"] or (args.with_covered and r["suspect"].startswith("marked as covered"))]
    extra: list[dict] = []
    ours = {r["plate_text"] for r in rows}
    for extra_csv in args.extra_csv:
        with extra_csv.open(encoding="utf-8", newline="") as handle:
            # A plate that is also one of ours would leak into the fold that holds it out.
            extra += [{"image_path": str((extra_csv.parent / r["image_path"]).resolve()), "plate_text": r["plate_text"]}
                      for r in csv.DictReader(handle) if r["plate_text"] and r["plate_text"] not in ours]
    extra.sort(key=lambda r: hashlib.sha256(Path(r["image_path"]).name.encode()).hexdigest())
    extra = extra[: args.extra_limit] if args.extra_limit else extra
    if args.final:
        train = [r for r in usable if not args.final_split or r["split"] in args.final_split] + extra
        args.out.mkdir(parents=True, exist_ok=True)
        say(args.out, f"final model: {args.epochs} epochs, lr {args.lr}"
                      + (f", crops of split {', '.join(args.final_split)} only" if args.final_split else ", all usable crops")
                      + (f", plus {len(extra)} crops from {', '.join(p.name for p in args.extra_csv)}" if extra else "")
                      + (f", weights averaged with the pretrained model ({args.blend_share:.2f} fine-tuned)" if args.blend_share < 1 else ""))
        model = fit(args.out, "final", train, args)
        say(args.out, f"RESULT  written {model}")
        return
    plans = []
    for fold in range(args.folds):
        train = [r for r in usable if fold_of(r["plate_text"], args.folds) != fold] + extra
        test = [r for r in rows if not r["suspect"] and r["source"] == "human" and fold_of(r["plate_text"], args.folds) == fold]
        if test and args.only_fold in (None, fold):
            plans.append((fold, train, test))
    args.out.mkdir(parents=True, exist_ok=True)
    say(args.out, f"cross-validation: {len(plans)} folds, {args.epochs} epochs each, lr {args.lr}, "
                  f"{'with' if args.with_covered else 'without'} covered plates"
                  + (f", plus {len(extra)} crops from {', '.join(p.name for p in args.extra_csv)}" if extra else "")
                  + (f", training on the {args.backend} backend" if args.backend != "tensorflow" else ""))
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(lambda plan: one_fold(*plan, args), plans))

    crops = [crop for result in results for crop in result["crops"]]
    total = len(crops)
    length = sum(len(c["label"]) for c in crops)
    summary = {
        "settings": {"folds": args.folds, "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
                     "weight_decay": args.weight_decay, "label_smoothing": args.label_smoothing,
                     "with_covered": args.with_covered, "seed": args.seed,
                     "extra_csv": [str(p) for p in args.extra_csv], "extra_crops": len(extra), "backend": args.backend,
                     "augmentation": str(args.augmentation) if args.augmentation else None,
                     "blend_share": args.blend_share, "baseline": str(args.baseline)},
        "crops": total,
        "plates": len({c["label"] for c in crops}),
        "current_right": sum(c["current"] == c["label"] for c in crops),
        "finetuned_right": sum(c["finetuned"] == c["label"] for c in crops),
        "fixed": sum(c["current"] != c["label"] and c["finetuned"] == c["label"] for c in crops),
        "broken": sum(c["current"] == c["label"] and c["finetuned"] != c["label"] for c in crops),
        "current_character_accuracy": round(1 - sum(c["current_distance"] for c in crops) / length, 4),
        "finetuned_character_accuracy": round(1 - sum(c["finetuned_distance"] for c in crops) / length, 4),
        "per_fold": [{"fold": r["fold"], "train_crops": r["train_crops"], "test_plates": r["test_plates"],
                      "test_crops": len(r["crops"]), "seconds": r["seconds"],
                      "current_right": sum(c["current"] == c["label"] for c in r["crops"]),
                      "finetuned_right": sum(c["finetuned"] == c["label"] for c in r["crops"])} for r in results],
    }
    if args.only_fold is None:
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    say(args.out, f"RESULT  {total} human-labelled crops of {summary['plates']} plates, each read by a model that never saw its plate")
    say(args.out, f"RESULT  baseline model: {summary['current_right']}/{total} = {summary['current_right'] / total:.1%} plates right, "
                  f"characters {summary['current_character_accuracy']:.1%}")
    say(args.out, f"RESULT  fine-tuned    : {summary['finetuned_right']}/{total} = {summary['finetuned_right'] / total:.1%} plates right, "
                  f"characters {summary['finetuned_character_accuracy']:.1%}")
    say(args.out, f"RESULT  crops it fixed: {summary['fixed']}   crops it broke: {summary['broken']}")
    for crop in crops:
        if crop["current"] == crop["label"] and crop["finetuned"] != crop["label"]:
            say(args.out, f"   broke  {crop['label']:<10} now read as {crop['finetuned'] or '(empty)':<10} {Path(crop['image']).name}")


if __name__ == "__main__":
    main()
