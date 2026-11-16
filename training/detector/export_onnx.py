"""Export an Ultralytics detector (.pt) to ONNX for the alpr-id runtime.

The runtime reads boxes as rows of x1, y1, x2, y2, confidence, class, so export with:
- ``--nms false`` for YOLO26: the end-to-end head, no suppression needed.
- ``--nms true`` for older models such as YOLOv8: suppression is built into the graph.

Ultralytics writes the .onnx next to the weights file. Always export from a copy inside this
project, never from a path in another project.

Usage (from the project root):
    uv run --project training python training/detector/export_onnx.py ^
        --weights models/detector/src/best.pt --nms true --out models/detector/plat.onnx
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

NMS_CHOICES = {"true": True, "false": False, "none": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--nms", choices=sorted(NMS_CHOICES), default="true")
    args = parser.parse_args()

    # Stop Ultralytics from installing packages on its own during export.
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    import onnxruntime as ort
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    print(f"names={model.names} task={model.task}")
    exported = Path(model.export(format="onnx", imgsz=args.imgsz, nms=NMS_CHOICES[args.nms]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != args.out.resolve():
        shutil.move(str(exported), str(args.out))

    session = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    print("inputs :", [(i.name, i.shape, i.type) for i in session.get_inputs()])
    print("outputs:", [(o.name, o.shape) for o in session.get_outputs()])
    print(f"saved {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
