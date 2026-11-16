"""Export a fine-tuned fast-plate-ocr model (.keras) to ONNX for the service.

This repeats what `fast-plate-ocr export --format onnx` does in version 1.1.0, with one
difference: that command writes through a NamedTemporaryFile that it reopens by name, which
Windows refuses (fast-plate-ocr issue #103). Here the intermediate file is a normal file in a
temporary folder.

The export runs on the TensorFlow backend so the outputs keep the names the service looks
for (`plate`, and `region` when the model has that head). The result takes the same input as
the hub model: uint8, batch x 64 x 128 x 3.

Usage (from the project root):
    uv run --project training/ocr python training/ocr/export_onnx.py \
        --model data/training/ocr/<run>/best.keras --plate-config data/training/ocr/<run>/plate_config.yaml \
        --out models/ocr/finetuned/<name>.onnx
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import onnx
import onnxruntime as ort
import onnxslim
from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
from fast_plate_ocr.train.utilities.utils import load_keras_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True, help="best.keras or last.keras of a training run")
    parser.add_argument("--plate-config", type=Path, required=True, help="plate_config.yaml of the same run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=None, help="default: the exporter's own choice")
    parser.add_argument("--blend-with", type=Path, default=None,
                        help="average the weights with those of this .keras file of the same architecture (the pretrained model)")
    parser.add_argument("--blend-share", type=float, default=0.5,
                        help="with --blend-with: share of --model in the average, the rest comes from --blend-with")
    args = parser.parse_args()

    plate_config = load_plate_config_from_yaml(args.plate_config)
    model = load_keras_model(args.model, plate_config)
    if args.blend_with:
        # A fine-tune on few crops fixes some reads and breaks others; halfway between the
        # pretrained and the fine-tuned weights keeps most of the fixes and few of the breaks
        # (measured by cross-validation, see docs). Only weights are read from the other file:
        # the hub's .keras was saved by a newer Keras and cannot be rebuilt as a model here.
        tuned = model.get_weights()
        model.load_weights(args.blend_with, skip_mismatch=True)
        other = model.get_weights()
        same = sum(bool(np.array_equal(a, b)) for a, b in zip(tuned, other, strict=True))
        model.set_weights([args.blend_share * a + (1 - args.blend_share) * b for a, b in zip(tuned, other, strict=True)])
        print(f"weights averaged: {args.blend_share:.2f} x {args.model.name} + {1 - args.blend_share:.2f} x {args.blend_with.name} "
              f"({same} of {len(tuned)} arrays are the same in both)")
    shape = (plate_config.img_height, plate_config.img_width, plate_config.num_channels)
    spec = [keras.InputSpec(name="input", shape=(None, *shape), dtype="uint8")]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        raw = str(Path(folder) / "model.onnx")
        model.export(raw, format="onnx", verbose=False, input_signature=spec, opset_version=args.opset)
        onnx.save(onnxslim.slim(onnx.load(raw)), str(args.out))

    # The exported file must answer like the Keras model it came from.
    session = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    sample = np.random.default_rng(0).integers(0, 256, size=(2, *shape), dtype=np.uint8)
    names = [output.name for output in session.get_outputs()]
    from_onnx = dict(zip(names, session.run(None, {session.get_inputs()[0].name: sample}), strict=True))
    from_keras = model.predict(sample, verbose=0)
    if not isinstance(from_keras, dict):
        raise SystemExit("expected the model to return named outputs (plate, region)")
    for name, expected in from_keras.items():
        if name not in from_onnx:
            raise SystemExit(f"output '{name}' is missing in the ONNX file; it has {names}")
        worst = float(np.abs(np.asarray(expected) - from_onnx[name]).max())
        if worst > 1e-3:
            raise SystemExit(f"output '{name}' differs from Keras by {worst:.4f}")
    described = ", ".join(f"{output.name} {output.shape}" for output in session.get_outputs())
    print(f"written {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")
    print(f"input {session.get_inputs()[0].name} {session.get_inputs()[0].type} {session.get_inputs()[0].shape}; outputs: {described}")
    print("ONNX and Keras agree on a random batch.")


if __name__ == "__main__":
    main()
