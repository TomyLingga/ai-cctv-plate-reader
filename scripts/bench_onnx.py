"""Time ONNX models on this machine, with the providers the service would use.

Meant for the inference PC: before a new detector or OCR model is installed, this shows
whether it loads on that GPU and how long one call takes next to the model in use.

    python scripts/bench_onnx.py models/detector/plat.onnx models/detector/candidates/yolo26n.onnx
    python scripts/bench_onnx.py models/ocr/cct_s_v2_global.onnx models/ocr/finetuned/new.onnx --batch 2

Inputs are random; only shape and type follow the model. Dimensions the model leaves open
are set to --batch (first) or 1.
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np
import onnxruntime as ort

from alpr_id.config import load_config
from alpr_id.detect.providers import resolve_providers
from alpr_id.logging_setup import configure_logging
from alpr_id.timing import clock

TYPES = {"tensor(float)": np.float32, "tensor(uint8)": np.uint8, "tensor(float16)": np.float16}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", type=Path, nargs="+")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    configure_logging(log_dir=None, level="ERROR")
    providers = resolve_providers(load_config().runtime)
    rng = np.random.default_rng(0)
    for path in args.models:
        session = ort.InferenceSession(str(path), providers=providers)
        feed = {}
        for index, item in enumerate(session.get_inputs()):
            shape = [dim if isinstance(dim, int) else (args.batch if position == 0 else 1) for position, dim in enumerate(item.shape)]
            kind = TYPES[item.type]
            feed[item.name] = (rng.integers(0, 256, size=shape, dtype=np.uint8) if kind is np.uint8
                               else rng.random(size=shape, dtype=np.float32).astype(kind))
            if index == 0:
                described = f"{item.name} {item.type} {shape}"
        for _ in range(20):
            session.run(None, feed)
        times = []
        for _ in range(args.runs):
            started = clock()
            session.run(None, feed)
            times.append((clock() - started) * 1000)
        outputs = ", ".join(f"{o.name} {o.shape}" for o in session.get_outputs())
        print(f"{path.name}: {session.get_providers()[0]}, input {described}, outputs {outputs}")
        print(f"    median {statistics.median(times):.2f} ms, p95 {sorted(times)[int(len(times) * 0.95) - 1]:.2f} ms, "
              f"max {max(times):.2f} ms over {args.runs} runs, file {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
