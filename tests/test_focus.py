"""The focus area: a second detector pass over the spot where vehicles stand.

Found on the weighbridge on 2026-11-17: a standing truck whose plate the detector did not
see at all in the wide search area, and saw with confidence 0.77 in a 640-pixel crop.
"""
import numpy as np
import pytest
from pydantic import ValidationError
from test_pipeline import STEP, ScriptedOcr

from alpr_id.config import AppConfig, CameraConfig, OcrConfig
from alpr_id.detect.onnx_detector import Detection
from alpr_id.pipeline.camera_pipeline import CameraPipeline, merge_detections
from alpr_id.timing import StageTimer

FRAME_SHAPE = (720, 1280, 3)
FOCUS = (0.25, 0.25, 0.75, 0.75)  # pixels 320..960 by 180..540
FOCUS_ORIGIN = (320.0, 180.0)
FOCUS_SIZE = (360, 640)  # height, width of the focus crop
PLATE_IN_FRAME = (500.0, 300.0, 660.0, 350.0)
PLATE_IN_FOCUS = (180.0, 120.0, 340.0, 170.0)  # the same plate in focus-crop pixels


class TwoViewDetector:
    """Returns the boxes scripted for the crop it is shown, told apart by the crop's size."""

    def __init__(self):
        self.wide: list[tuple[tuple[float, float, float, float], float]] = []
        self.focus: list[tuple[tuple[float, float, float, float], float]] = []
        self.calls = 0

    def preprocess(self, image):
        return image, None

    def infer(self, tensor):
        self.calls += 1

    def postprocess(self, output, box, shape):
        boxes = self.focus if tuple(shape[:2]) == FOCUS_SIZE else self.wide
        return [Detection(*b, confidence) for b, confidence in boxes]


class TwoViewHarness:
    def __init__(self, focus=FOCUS):
        camera = CameraConfig(id="cam-1", legacy_id=1, url_env="X", focus=focus, process_fps=8)
        config = AppConfig(cameras=(camera,), ocr=OcrConfig(read_when="always"))
        self.detector = TwoViewDetector()
        self.events = []
        self.pipeline = CameraPipeline(camera, config, self.detector, ScriptedOcr(), StageTimer(), on_event=self._on_event)
        self.t = 0.0
        self._rng = np.random.default_rng(0)

    def _on_event(self, event, crop, frame):
        self.events.append(event)
        return event

    def frames(self, count: int):
        result = None
        for _ in range(count):
            image = self._rng.integers(0, 255, FRAME_SHAPE, dtype=np.uint8)
            result = self.pipeline.process(image, self.t, 1_794_000_000.0 + self.t)
            self.t += STEP
        return result


def test_plate_seen_only_in_the_focus_area_is_read():
    h = TwoViewHarness()
    h.detector.focus = [(PLATE_IN_FOCUS, 0.8)]
    result = h.frames(16)
    assert [e.plate for e in h.events] == ["BK1234MA"]
    # Boxes are in frame pixels, not in pixels of the focus crop. The event's box is the
    # crop that was read, which is the plate plus a little padding.
    assert result.visible[0].box == tuple(int(v) for v in PLATE_IN_FRAME)
    x1, y1, x2, y2 = h.events[0].bbox
    assert 480 < x1 <= 500 and 290 < y1 <= 300 and 660 <= x2 < 680 and 350 <= y2 < 360
    assert len(result.visible) == 1


def test_plate_seen_in_both_views_is_one_vehicle():
    h = TwoViewHarness()
    h.detector.wide = [(PLATE_IN_FRAME, 0.6)]
    h.detector.focus = [((181.0, 121.0, 339.0, 169.0), 0.8)]  # same plate, box off by a pixel
    result = h.frames(16)
    assert len(h.events) == 1
    assert len(result.visible) == 1


def test_plate_cut_by_the_focus_border_is_left_to_the_wide_view():
    h = TwoViewHarness()
    h.detector.focus = [((0.0, 120.0, 90.0, 170.0), 0.9)]  # touches the left border of the focus crop
    result = h.frames(16)
    assert h.events == []
    assert result.visible == ()


def test_focus_pass_only_looks_in_now_and_then_while_nothing_is_in_view():
    without, idle = TwoViewHarness(focus=None), TwoViewHarness()
    without.frames(8)
    idle.frames(8)
    # Eight frames: eight wide passes, plus the focus pass on every fourth frame.
    assert (without.detector.calls, idle.detector.calls) == (8, 10)


def test_focus_pass_runs_on_every_frame_while_a_plate_is_in_view_and_a_while_after():
    h = TwoViewHarness()
    h.detector.wide = [(PLATE_IN_FRAME, 0.9)]
    h.frames(8)
    # The first frame is still idle (wide pass only); it finds the plate, so the other seven get both.
    assert h.detector.calls == 8 + 7
    h.detector.wide = []
    h.detector.calls = 0
    h.frames(48)  # six seconds with nothing in view: both passes for three seconds, then idle again
    assert h.detector.calls == 48 + 24 + 6


def test_merge_keeps_the_more_confident_box_of_the_same_plate():
    wide = [Detection(100, 100, 200, 130, 0.4), Detection(600, 300, 700, 330, 0.9)]
    focus = [Detection(102, 101, 201, 131, 0.8), Detection(600, 300, 700, 330, 0.5), Detection(900, 500, 990, 530, 0.7)]
    merged = merge_detections(wide, focus, 0.5)
    assert [(d.x1, d.confidence) for d in merged] == [(102, 0.8), (600, 0.9), (900, 0.7)]


def test_focus_must_be_a_valid_area_inside_roi():
    assert CameraConfig(id="cam-1").focus is None
    CameraConfig(id="cam-1", roi=(0.1, 0.1, 0.9, 0.9), focus=(0.3, 0.3, 0.6, 0.6))
    with pytest.raises(ValidationError, match="focus must lie inside roi"):
        CameraConfig(id="cam-1", roi=(0.2, 0.2, 0.8, 0.8), focus=(0.1, 0.3, 0.6, 0.6))
    with pytest.raises(ValidationError):
        CameraConfig(id="cam-1", focus=(0.6, 0.3, 0.3, 0.6))
