"""Write the augmentation used for fine-tuning: the trainer's default, pushed towards CCTV crops.

Public plate crops are sharp close-ups. A plate on the weighbridge cameras is about 100 to
130 pixels wide, often soft and compressed. Starting from the pipeline the trainer saves with
every run (train_augmentation.yaml), this raises how often and how strongly an image is
scaled down, blurred and JPEG-compressed, so sharp crops also teach the model to read soft
ones. Erosion and dilation are switched off: on a 128 pixel wide plate they can close the
short middle stroke that tells M and W from H.

    uv run --project training/ocr python training/ocr/make_augmentation.py \
        --base data/training/ocr/<run>/train_augmentation.yaml --out training/ocr/augmentation-cctv.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import albumentations as A
import numpy as np
import yaml


def walk(node: dict, seen: list[str]) -> None:
    name = node.get("__class_fullname__", "")
    children = node.get("transforms", [])
    kinds = {child.get("__class_fullname__") for child in children}
    if name == "Downscale":
        node["p"], node["scale_range"] = 0.45, [0.45, 0.95]
        seen.append(name)
    elif name == "GaussianBlur":
        node["sigma_limit"] = [0.3, 1.2]
        seen.append(name)
    elif name == "MotionBlur":
        node["blur_limit"] = [3, 5]
        seen.append(name)
    elif name == "ImageCompression":
        node["p"], node["quality_range"] = 0.3, [30, 85]
        seen.append(name)
    elif name == "OneOf" and kinds == {"GaussianBlur", "MotionBlur"}:
        node["p"] = 0.35
        seen.append("OneOf blur")
    elif name == "OneOf" and "ImageCompression" in kinds:
        node["p"] = 0.35
        seen.append("OneOf noise")
    elif name == "OneOf" and kinds == {"Morphological"}:
        node["p"] = 0.0
        seen.append("OneOf morphological")
    for child in children:
        walk(child, seen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, required=True, help="train_augmentation.yaml of any training run")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    seen: list[str] = []
    walk(data["transform"], seen)
    expected = {"Downscale", "GaussianBlur", "MotionBlur", "ImageCompression", "OneOf blur", "OneOf noise", "OneOf morphological"}
    if set(seen) != expected:
        raise SystemExit(f"the base pipeline changed: found {sorted(seen)}, expected {sorted(expected)}")
    args.out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    # The file must load and run the way the trainer will use it.
    pipeline = A.load(str(args.out), data_format="yaml")
    sample = np.random.default_rng(0).integers(0, 256, size=(64, 128, 3), dtype=np.uint8)
    for _ in range(50):
        result = pipeline(image=sample)["image"]
        if result.shape != sample.shape:
            raise SystemExit(f"augmented image has shape {result.shape}")
    print(f"written {args.out}: {', '.join(sorted(seen))} changed")


if __name__ == "__main__":
    main()
