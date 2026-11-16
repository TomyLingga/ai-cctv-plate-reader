"""Compare the fold models of cross-validation runs on the crops they never saw, per crop and per vehicle.

crossval.py prints how many crops a fine-tune reads right. This report adds what decides whether
a model may replace the one in the service:

  fix / broke      crops the baseline model read wrong and this one right, and the other way round
  wrong >= 0.9     wrong reads the model is sure about: those become wrong events
  veh              vehicles read right when all crops of a vehicle go through the service's own
                   voter (alpr_id.postprocess.voting.PlateVoter), no correction applied
  veh2+            the same for vehicles with at least two crops

A run is NAME=<cv folder>, or NAME=<cv folder>@<file> for another model file in each fold folder
(for example a fold model whose weights were averaged with the pretrained ones by
export_onnx.py --blend-with). With --ensembles N the report also averages the probabilities of
up to N models, the baseline included, as a runtime ensemble would.

Usage (from the project root):
    uv run --frozen --extra cpu python training/ocr/cv_report.py \
        --run G2=data/training/ocr/cv-20261115-g2 --run G2w50=data/training/ocr/cv-20261115-g2@blend-G2w50.onnx --detail G2w50
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from alpr_id.config import load_config
from alpr_id.detect.providers import resolve_providers
from alpr_id.logging_setup import configure_logging
from alpr_id.ocr.recognizer import PlateOcr
from alpr_id.postprocess.voting import OcrRead, PlateVoter

CROP_NAME = re.compile(r"^\d{6}_t(\d+)_")  # the service names a saved crop <time>_t<track>_<read>_...


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, help="NAME=<cv folder>[@<model file in each fold folder>]")
    parser.add_argument("--ensembles", type=int, default=1, help="also average the probabilities of up to this many models")
    parser.add_argument("--detail", default=None, help="variant to list broken crops, changed vehicles and substitutions for")
    parser.add_argument("--baseline", type=Path, default=Path("models/ocr/cct_s_v2_global.onnx"),
                        help="the model every variant is compared with, listed as 'baseline'; keep it the hub model, "
                             "because a model trained on all labelled crops has seen the plates of every fold")
    args = parser.parse_args()

    configure_logging(log_dir=None, level="ERROR")
    config = load_config()
    providers = resolve_providers(config.runtime)
    plate_config = config.ocr.plate_config_path
    baseline = PlateOcr(args.baseline, plate_config, providers)
    alphabet, pad = baseline.alphabet, baseline.pad_char

    runs = {}
    for spec in args.run:
        name, where = spec.split("=", 1)
        folder, _, file = where.partition("@")
        runs[name] = (Path(folder), file or "model.onnx")
    first = next(iter(runs.values()))[0]

    crops = []
    for fold in sorted(first.glob("fold-*")):
        models = {name: PlateOcr(folder / fold.name / file, plate_config, providers) for name, (folder, file) in runs.items()}
        with (fold / "test.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            path = (fold / row["image_path"]).resolve()
            image = cv2.imread(str(path))
            if image is None:
                raise SystemExit(f"cannot read {path}")
            track = CROP_NAME.match(path.name).group(1)
            probs = {"baseline": baseline.read([image])[0].probs}
            probs.update({name: model.read([image])[0].probs for name, model in models.items()})
            # date, camera, track and plate together are one visit of one vehicle to one camera
            crops.append({"label": row["plate_text"], "name": path.name, "probs": probs,
                          "vehicle": (path.parts[-4], path.parts[-3], int(track), row["plate_text"])})

    names = ["baseline", *runs]
    variants = {name: (name,) for name in names}
    for size in range(2, min(len(names), args.ensembles) + 1):
        for members in combinations(names, size):
            variants["+".join(members)] = members

    def read_of(crop: dict, members: tuple[str, ...]) -> OcrRead:
        return OcrRead.from_probs(np.mean([crop["probs"][m] for m in members], axis=0), alphabet, pad)

    vehicles = defaultdict(list)
    for crop in crops:
        vehicles[crop["vehicle"]].append(crop)
    several = [key for key, items in vehicles.items() if len(items) >= 2]

    def voted(items: list[dict], members: tuple[str, ...]) -> str:
        voter = PlateVoter(alphabet, pad)
        for crop in items:
            voter.add(read_of(crop, members))
        result = voter.result()
        return result.text if result else ""

    table = {}
    for label, members in variants.items():
        reads = [read_of(crop, members) for crop in crops]
        table[label] = {"reads": reads, "right": [r.text == c["label"] for r, c in zip(reads, crops, strict=True)],
                        "votes": {key: voted(items, members) for key, items in vehicles.items()}}

    base = table["baseline"]
    print(f"{len(crops)} human-labelled crops of {len({c['label'] for c in crops})} plates, {len(vehicles)} vehicles, "
          f"{len(several)} of them with at least two crops; every model reads only plates it never saw")
    print(f"{'variant':<24} {'crops':>5} {'fix':>4} {'broke':>5} {'wrong>=0.9':>10} {'veh':>4} {'veh2+':>5} {'veh fix/broke':>13}")
    for label, row in table.items():
        wrong_sure = sum(r.confidence >= 0.9 for r, ok in zip(row["reads"], row["right"], strict=True) if not ok)
        fixed = sum(b and not a for a, b in zip(base["right"], row["right"], strict=True))
        broke = sum(a and not b for a, b in zip(base["right"], row["right"], strict=True))
        good = {key: text == key[3] for key, text in row["votes"].items()}
        was = {key: text == key[3] for key, text in base["votes"].items()}
        print(f"{label:<24} {sum(row['right']):>5} {fixed:>4} {broke:>5} {wrong_sure:>10} {sum(good.values()):>4} "
              f"{sum(good[key] for key in several):>5} {sum(good[k] and not was[k] for k in vehicles):>8}/{sum(was[k] and not good[k] for k in vehicles)}")

    if not args.detail:
        return
    if args.detail not in table:
        raise SystemExit(f"--detail must be one of: {', '.join(table)}")
    row = table[args.detail]
    members = variants[args.detail]
    print(f"\n{args.detail}: crops it broke")
    for crop, was_right, read in zip(crops, base["right"], row["reads"], strict=True):
        if was_right and read.text != crop["label"]:
            print(f"   {crop['label']:<10} now {read.text or '(empty)':<10} conf {read.confidence:.2f}  {crop['name']}")
    print(f"\n{args.detail}: vehicles whose voted plate changed between right and wrong")
    for key, items in vehicles.items():
        before, after = base["votes"][key], row["votes"][key]
        if (before == key[3]) != (after == key[3]):
            texts = ",".join(read_of(crop, members).text or "-" for crop in items)
            print(f"   {'fixed' if after == key[3] else 'broke'}  {key[3]:<10} {key[0]} {key[1]} t{key[2]}: was {before or '-'}, now {after or '-'}  [{texts}]")
    print("\nmost frequent substitutions in wrong reads of the right length (label->read)")
    for label in ("baseline", args.detail):
        swaps: Counter[tuple[str, str]] = Counter()
        for crop, read in zip(crops, table[label]["reads"], strict=True):
            if read.text != crop["label"] and len(read.text) == len(crop["label"]):
                swaps.update((a, b) for a, b in zip(crop["label"], read.text, strict=True) if a != b)
        print(f"   {label:<10} " + ", ".join(f"{a}->{b} x{n}" for (a, b), n in swaps.most_common(12)))


if __name__ == "__main__":
    main()
