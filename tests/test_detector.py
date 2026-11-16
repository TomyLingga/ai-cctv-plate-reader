import numpy as np
import pytest

from alpr_id.capture.roi import clamp_box, roi_to_pixels
from alpr_id.detect.onnx_detector import PlateDetector, decode_output, letterbox


class FakeInput:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


class FakeSession:
    """Stands in for an ONNX Runtime session that returns a prepared output."""

    def __init__(self, output, input_shape=(1, 3, 640, 640)):
        self._output = output
        self._input_shape = list(input_shape)
        self.fed = None

    def get_inputs(self):
        return [FakeInput("images", self._input_shape)]

    def get_outputs(self):
        return [FakeInput("output0", list(self._output.shape))]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _names, feed):
        self.fed = feed
        return [self._output]


def test_letterbox_keeps_aspect_ratio_and_centres_the_image():
    image = np.full((648, 1280, 3), 200, dtype=np.uint8)
    canvas, box = letterbox(image, 640, 640)
    assert canvas.shape == (640, 640, 3)
    assert box.scale == pytest.approx(0.5)
    assert (box.pad_x, box.pad_y) == (0.0, 158.0)
    assert canvas[0, 0, 0] == 114  # padding above
    assert canvas[320, 320, 0] == 200  # image in the middle


def test_decode_rows_of_xyxy_conf_class():
    output = np.zeros((1, 300, 6), dtype=np.float32)
    output[0, 0] = [100, 200, 180, 230, 0.9, 0]
    output[0, 1] = [300, 300, 340, 320, 0.2, 0]
    rows = decode_output(output, conf_threshold=0.35, iou_threshold=0.5)
    assert rows.shape == (1, 5)
    assert rows[0].tolist() == pytest.approx([100, 200, 180, 230, 0.9])


def test_decode_yolov9_end2end_rows():
    # batch index, x1, y1, x2, y2, class, score
    output = np.array([[0, 100, 200, 180, 230, 0, 0.8], [0, 10, 10, 20, 20, 0, 0.1], [1, 5, 5, 9, 9, 0, 0.99]], dtype=np.float32)
    rows = decode_output(output, conf_threshold=0.35, iou_threshold=0.5)
    assert rows.shape == (1, 5)
    assert rows[0].tolist() == pytest.approx([100, 200, 180, 230, 0.8])


def test_decode_raw_output_runs_suppression():
    anchors = 8400
    output = np.zeros((1, 5, anchors), dtype=np.float32)
    # Two nearly identical boxes (centre x, centre y, width, height, score) and one elsewhere.
    output[0, :, 0] = [140, 215, 80, 30, 0.90]
    output[0, :, 1] = [141, 215, 80, 30, 0.85]
    output[0, :, 2] = [400, 400, 60, 24, 0.70]
    rows = decode_output(output, conf_threshold=0.35, iou_threshold=0.5)
    assert rows.shape == (2, 5)
    best = rows[rows[:, 4].argmax()]
    assert best.tolist() == pytest.approx([100, 200, 180, 230, 0.90])


def test_decode_rejects_unknown_layout():
    with pytest.raises(ValueError):
        decode_output(np.zeros((1, 2, 3, 4), dtype=np.float32), 0.3, 0.5)


def test_detector_maps_boxes_back_to_the_image():
    output = np.zeros((1, 300, 6), dtype=np.float32)
    # In the 640x640 input of a 1280x648 image: scale 0.5, 158 px of padding above.
    output[0, 0] = [100, 258, 160, 278, 0.8, 0]
    session = FakeSession(output)
    detector = PlateDetector(session, conf_threshold=0.35)
    image = np.zeros((648, 1280, 3), dtype=np.uint8)
    detections = detector.detect(image)
    assert session.fed["images"].shape == (1, 3, 640, 640)
    assert session.fed["images"].dtype == np.float32
    assert len(detections) == 1
    assert detections[0].box == pytest.approx((200.0, 200.0, 320.0, 240.0))
    assert detections[0].shifted(10, 20).box == pytest.approx((210.0, 220.0, 330.0, 260.0))


def test_detector_requires_a_fixed_input_size():
    with pytest.raises(ValueError):
        PlateDetector(FakeSession(np.zeros((1, 300, 6), dtype=np.float32), input_shape=(1, 3, "height", "width")))


def test_roi_to_pixels_and_clamp_box():
    roi = roi_to_pixels((0.25, 0.5, 0.75, 1.0), 2560, 1440)
    assert (roi.x1, roi.y1, roi.x2, roi.y2) == (640, 720, 1920, 1440)
    assert (roi.width, roi.height) == (1280, 720)
    rect = clamp_box((100.0, 100.0, 200.0, 140.0), 2560, 1440, padding=0.1)
    assert (rect.x1, rect.y1, rect.x2, rect.y2) == (90, 96, 210, 144)
    assert clamp_box((2559.5, 10.0, 2560.0, 30.0), 2560, 1440) is None
