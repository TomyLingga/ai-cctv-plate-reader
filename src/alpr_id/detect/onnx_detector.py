"""Plate detector on ONNX Runtime.

Reads the three output layouts used by the detectors of this project:

- ``(N, 300, 6)``: rows of x1, y1, x2, y2, confidence, class. Ultralytics export with the
  end-to-end head (YOLO26, ``nms=False``) or with NMS built in (``nms=True``).
- ``(N, 4 + classes, anchors)``: centre x, centre y, width, height, class scores. Ultralytics
  export without NMS; suppression is done here.
- ``(M, 7)``: rows of batch index, x1, y1, x2, y2, class, score. YOLOv9 end-to-end export
  used by open-image-models.

All three give boxes in pixels of the letterboxed network input, which are mapped back to
the image that was passed in.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

PAD_VALUE = 114


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def shifted(self, dx: float, dy: float) -> Detection:
        return Detection(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy, self.confidence)


@dataclass(frozen=True)
class Letterbox:
    scale: float
    pad_x: float
    pad_y: float


def letterbox(image: np.ndarray, height: int, width: int) -> tuple[np.ndarray, Letterbox]:
    """Resize keeping the aspect ratio and pad to (height, width), image centred."""
    src_h, src_w = image.shape[:2]
    scale = min(height / src_h, width / src_w)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
    resized = image if (new_w, new_h) == (src_w, src_h) else cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (width - new_w) / 2, (height - new_h) / 2
    left, top = int(round(pad_x - 0.1)), int(round(pad_y - 0.1))
    canvas = np.full((height, width, 3), PAD_VALUE, dtype=np.uint8)
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, Letterbox(scale, float(left), float(top))


def decode_output(output: np.ndarray, conf_threshold: float, iou_threshold: float) -> np.ndarray:
    """Rows of x1, y1, x2, y2, confidence in network-input pixels, for the first image."""
    if output.ndim == 2 and output.shape[1] == 7:
        rows = output[output[:, 0] == 0]
        boxes = np.concatenate([rows[:, 1:5], rows[:, 6:7]], axis=1)
        return boxes[boxes[:, 4] >= conf_threshold]
    if output.ndim == 3 and output.shape[2] == 6:
        rows = output[0]
        return rows[rows[:, 4] >= conf_threshold][:, :5]
    if output.ndim == 3 and output.shape[1] < output.shape[2]:
        rows = output[0].T  # (anchors, 4 + classes)
        scores = rows[:, 4:].max(axis=1)
        keep = scores >= conf_threshold
        rows, scores = rows[keep], scores[keep]
        if rows.shape[0] == 0:
            return np.empty((0, 5), dtype=np.float32)
        top_left = np.stack([rows[:, 0] - rows[:, 2] / 2, rows[:, 1] - rows[:, 3] / 2, rows[:, 2], rows[:, 3]], axis=1)
        picked = cv2.dnn.NMSBoxes(top_left.tolist(), scores.tolist(), conf_threshold, iou_threshold)
        picked = np.asarray(picked, dtype=int).reshape(-1)
        xyxy = np.stack(
            [top_left[:, 0], top_left[:, 1], top_left[:, 0] + top_left[:, 2], top_left[:, 1] + top_left[:, 3]], axis=1
        )
        return np.concatenate([xyxy[picked], scores[picked, None]], axis=1)
    raise ValueError(f"Unsupported detector output shape {output.shape}")


class PlateDetector:
    def __init__(self, session: ort.InferenceSession, conf_threshold: float = 0.35, iou_threshold: float = 0.5) -> None:
        self._session = session
        model_input = session.get_inputs()[0]
        self._input_name = model_input.name
        shape = model_input.shape
        if len(shape) != 4 or not all(isinstance(dim, int) for dim in shape[2:]):
            raise ValueError(f"Detector input must be (N, 3, H, W) with fixed H and W, got {shape}")
        self.input_height, self.input_width = int(shape[2]), int(shape[3])
        self._output_name = session.get_outputs()[0].name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    @property
    def providers(self) -> list[str]:
        return self._session.get_providers()

    def preprocess(self, image_bgr: np.ndarray) -> tuple[np.ndarray, Letterbox]:
        canvas, box = letterbox(image_bgr, self.input_height, self.input_width)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
        return tensor, box

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        return self._session.run([self._output_name], {self._input_name: tensor})[0]

    def postprocess(self, output: np.ndarray, box: Letterbox, image_shape: tuple[int, ...]) -> list[Detection]:
        rows = decode_output(output, self.conf_threshold, self.iou_threshold)
        height, width = image_shape[:2]
        detections: list[Detection] = []
        for x1, y1, x2, y2, score in rows:
            dx1 = min(max((float(x1) - box.pad_x) / box.scale, 0.0), width)
            dy1 = min(max((float(y1) - box.pad_y) / box.scale, 0.0), height)
            dx2 = min(max((float(x2) - box.pad_x) / box.scale, 0.0), width)
            dy2 = min(max((float(y2) - box.pad_y) / box.scale, 0.0), height)
            if dx2 - dx1 >= 2 and dy2 - dy1 >= 2:
                detections.append(Detection(dx1, dy1, dx2, dy2, float(score)))
        return detections

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        tensor, box = self.preprocess(image_bgr)
        return self.postprocess(self.infer(tensor), box, image_bgr.shape)
