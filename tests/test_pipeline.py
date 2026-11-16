"""Event logic of the camera pipeline, with a scripted detector and OCR."""
import numpy as np
import pytest

from alpr_id.config import AppConfig, CameraConfig, OcrConfig, VotingConfig
from alpr_id.detect.onnx_detector import Detection
from alpr_id.pipeline.camera_pipeline import CameraPipeline
from alpr_id.postprocess.voting import OcrRead
from alpr_id.timing import StageTimer

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
FRAME_SHAPE = (720, 1280, 3)
STEP = 0.125  # 8 frames per second
CENTRE_BOX = (500.0, 300.0, 660.0, 350.0)


def make_read(text: str, confidence: float) -> OcrRead:
    probs = np.full((10, len(ALPHABET)), (1.0 - confidence) / (len(ALPHABET) - 1), dtype=np.float32)
    for slot in range(10):
        probs[slot, ALPHABET.index(text[slot] if slot < len(text) else "_")] = confidence
    return OcrRead.from_probs(probs, ALPHABET, "_")


def centred_box(width: float) -> tuple[float, float, float, float]:
    height = width / 3.2
    return (640 - width / 2, 360 - height / 2, 640 + width / 2, 360 + height / 2)


class ScriptedDetector:
    def __init__(self):
        self.boxes: list[tuple[float, float, float, float]] = []

    def preprocess(self, image):
        return image, None

    def infer(self, tensor):
        return None

    def postprocess(self, output, box, shape):
        return [Detection(*b, 0.9) for b in self.boxes]


class ScriptedOcr:
    """Returns the same text for every crop unless `script` decides per crop."""

    alphabet = ALPHABET
    pad_char = "_"

    def __init__(self, text="BK1234MA", confidence=0.97, script=None):
        self.text, self.confidence = text, confidence
        self.script = script  # callable(read_number, crop) -> (text, confidence)
        self.crops_read = 0

    def read(self, crops):
        reads = []
        for crop in crops:
            self.crops_read += 1
            text, confidence = self.script(self.crops_read, crop) if self.script else (self.text, self.confidence)
            reads.append(make_read(text, confidence))
        return reads


class Harness:
    def __init__(self, fake_ocr: ScriptedOcr | None = None, **overrides):
        # These tests describe how reads are spent once reading is allowed. Waiting for the
        # vehicle to stop is off unless a test asks for it (tests/test_stop_and_region.py).
        overrides.setdefault("ocr", OcrConfig(read_when="always"))
        camera = CameraConfig(id="cam-1", legacy_id=1, url_env="X", roi=(0.0, 0.0, 1.0, 1.0), process_fps=8)
        self.config = AppConfig(cameras=(camera,), **overrides)
        self.detector = ScriptedDetector()
        self.ocr = fake_ocr or ScriptedOcr()
        self.events = []
        self.reads = []
        self.read_times = []
        self.pipeline = CameraPipeline(
            camera, self.config, self.detector, self.ocr, StageTimer(),
            on_event=self._on_event, on_read=self._on_read,
        )
        self.t = 0.0
        self._rng = np.random.default_rng(0)

    def _on_event(self, event, crop, frame):
        self.events.append(event)
        return event.with_media("/media/crop.jpg", "/media/frame.jpg")

    def _on_read(self, *args):
        self.reads.append(args)
        self.read_times.append(self.t)

    def frame(self, box=CENTRE_BOX):
        self.detector.boxes = [box] if box else []
        image = self._rng.integers(0, 255, FRAME_SHAPE, dtype=np.uint8)
        result = self.pipeline.process(image, self.t, 1_794_000_000.0 + self.t)
        self.t += STEP
        return result

    def frames(self, count: int, box=CENTRE_BOX):
        result = None
        for _ in range(count):
            result = self.frame(box)
        return result

    @property
    def now(self) -> float:
        return self.t - STEP


def test_one_event_per_vehicle_however_long_it_stays():
    h = Harness()
    h.frames(80)  # ten seconds standing on the scale
    assert len(h.events) == 1
    event = h.events[0]
    assert (event.plate, event.plate_formatted) == ("BK1234MA", "BK 1234 MA")
    assert event.format_valid and event.region_known
    assert event.confidence == pytest.approx(0.97, abs=0.01)
    assert event.camera_id == "cam-1"
    assert event.timestamp.endswith("+07:00")
    assert event.bbox[2] > event.bbox[0]


def test_ocr_stops_after_the_event_and_never_exceeds_the_limit():
    h = Harness()
    h.frames(100)
    # Confident and valid after two reads, so no further OCR is spent on this vehicle.
    assert h.ocr.crops_read == 2
    assert len(h.reads) == 2

    hesitant = Harness(fake_ocr=ScriptedOcr(confidence=0.70))
    hesitant.frames(160)
    assert hesitant.ocr.crops_read == hesitant.config.ocr.max_reads_per_track == 5


def test_first_reads_come_quickly_and_the_rest_are_spread_out():
    h = Harness(fake_ocr=ScriptedOcr(confidence=0.70))
    h.frames(100)
    # Two reads within the first second, then one every stale_read_s while nothing improves.
    assert h.read_times[1] < 1.0
    assert h.read_times[2] - h.read_times[1] >= h.config.ocr.stale_read_s
    assert h.read_times[4] > 9.0


def test_reads_are_kept_for_better_crops_as_the_vehicle_approaches():
    # Small crops read badly, crops from 150 px on read well.
    def script(_number, crop):
        return ("BK1284MA", 0.60) if crop.shape[1] < 150 else ("BK1234MA", 0.97)

    h = Harness(fake_ocr=ScriptedOcr(script=script))
    for width in np.linspace(80, 240, 48):  # six seconds of approach
        h.frame(centred_box(float(width)))
    h.frames(40, box=centred_box(240.0))
    assert len(h.events) == 1
    # With all five reads spent in the first 1.6 seconds the result would have been BK1284MA.
    assert h.events[0].plate == "BK1234MA"
    widths = [read[2].shape[1] for read in h.reads]
    assert widths == sorted(widths) and widths[-1] >= 150


def test_uncertain_result_is_emitted_once_the_reads_are_used_up():
    h = Harness(fake_ocr=ScriptedOcr(confidence=0.70))
    h.frames(100)
    assert len(h.events) == 1
    assert h.events[0].confidence == pytest.approx(0.70, abs=0.01)
    assert h.events[0].reads == 5


def test_one_uncertain_character_holds_back_the_early_event():
    # The reads disagree on the fifth character; the average confidence alone would pass.
    def script(number, _crop):
        return ("BK1284AB", 0.96) if number == 2 else ("BK1234AB", 0.97)

    h = Harness(fake_ocr=ScriptedOcr(script=script))
    h.frames(16)
    assert h.ocr.crops_read == 2 and h.events == []
    h.frames(40)
    assert len(h.events) == 1
    assert h.events[0].plate == "BK1234AB" and h.events[0].reads == 3


def test_result_too_weak_is_dropped_and_not_shown():
    h = Harness(fake_ocr=ScriptedOcr(confidence=0.20))
    h.frames(100)
    assert h.events == []
    assert h.pipeline.current(h.now) is None
    assert h.pipeline.last_plate() is None
    h.pipeline.flush()
    assert h.events == []


def test_vehicle_that_read_badly_at_first_is_read_again_while_it_stays():
    def script(number, _crop):
        return ("BK1284NA", 0.20) if number <= 5 else ("BK1234MA", 0.97)

    h = Harness(fake_ocr=ScriptedOcr(script=script))
    h.frames(100)
    assert h.events == [] and h.ocr.crops_read == 5
    h.frames(80)  # still on the scale after the retry pause
    assert len(h.events) == 1
    assert h.events[0].plate == "BK1234MA"
    assert h.ocr.crops_read == 7


def test_no_second_attempt_when_retry_is_switched_off():
    h = Harness(fake_ocr=ScriptedOcr(confidence=0.20), ocr=OcrConfig(retry_after_s=0, read_when="always"))
    h.frames(240)
    assert h.ocr.crops_read == 5


def test_plate_outside_the_pattern_is_reported_with_flag():
    h = Harness(fake_ocr=ScriptedOcr(text="8039200", confidence=0.95))
    h.frames(100)
    assert len(h.events) == 1
    assert h.events[0].plate == "8039200"
    assert h.events[0].format_valid is False


def test_same_plate_under_a_new_track_is_the_same_vehicle():
    h = Harness()
    h.frames(30)
    h.frames(60, box=None)  # the plate is hidden long enough for the track to close
    h.frames(30)  # it reappears under a new track id
    assert len(h.events) == 1


def test_dedup_window_counts_from_when_the_vehicle_was_last_seen():
    h = Harness(voting=VotingConfig(dedup_window_s=10.0))
    h.frames(96)  # twelve seconds on the scale; the event came in the first second
    h.frames(60, box=None)  # hidden for 7.5 seconds, the track closes
    h.frames(30)
    assert len(h.events) == 1


def test_new_track_id_at_the_same_spot_continues_the_same_vehicle():
    h = Harness()
    h.frames(20)
    first = h.pipeline.current(h.now)
    assert len(h.events) == 1 and h.ocr.crops_read == 2

    h.frames(36, box=None)  # hidden for 4.5 s: the tracker drops the id, the vehicle state is still open
    h.ocr.text = "BK1234HA"  # what a fresh read of the same plate would give now
    h.frames(20)

    current = h.pipeline.current(h.now)
    assert current.track_id != first.track_id  # the tracker did start a new track
    assert current.plate == "BK1234MA" and current.event_id == h.events[0].event_id
    assert len(h.events) == 1
    assert h.ocr.crops_read == 2  # the vehicle was not read again


def test_same_plate_after_the_window_is_a_new_event():
    h = Harness(voting=VotingConfig(dedup_window_s=10.0))
    h.frames(30)
    h.frames(120, box=None)  # fifteen seconds without the plate
    h.frames(30)
    assert len(h.events) == 2


def test_plate_cut_by_the_search_area_is_not_read():
    h = Harness()
    h.frames(40, box=(0.0, 300.0, 160.0, 350.0))  # touches the left border
    assert h.ocr.crops_read == 0
    assert h.events == []


def test_plate_narrower_than_the_minimum_is_not_read():
    h = Harness()
    h.frames(40, box=(500.0, 300.0, 540.0, 315.0))  # 40 px wide, minimum is 60
    assert h.ocr.crops_read == 0


def test_track_that_ends_before_it_is_stable_emits_on_close():
    h = Harness(fake_ocr=ScriptedOcr(confidence=0.70))
    h.frames(4)  # confirmed after two frames, one or two reads only
    assert h.events == []
    h.frames(60, box=None)
    assert len(h.events) == 1


def test_tracks_time_out_even_when_no_frames_arrive():
    h = Harness(fake_ocr=ScriptedOcr(confidence=0.70))
    h.frames(4)
    assert h.events == []
    closed = h.pipeline.expire(h.now + 10.0)  # the camera went silent
    assert len(closed) == 1 and len(h.events) == 1
    assert h.pipeline.visible() == ()


def test_flush_closes_open_tracks():
    h = Harness(fake_ocr=ScriptedOcr(confidence=0.70))
    h.frames(4)
    assert h.events == []
    flushed = h.pipeline.flush()
    assert len(flushed) == 1 and len(h.events) == 1


def test_failing_event_sink_does_not_stop_the_camera_or_repeat_the_event():
    h = Harness()
    calls = []

    def failing_sink(event, crop, frame):
        calls.append(event.event_id)
        raise OSError("disk full")

    h.pipeline._on_event = failing_sink
    h.frames(40)  # must not raise
    assert len(calls) == 1
    current = h.pipeline.current(h.now)
    assert current is not None and current.plate == "BK1234MA"
    assert current.event_id == calls[0]


def test_current_plate_is_visible_while_the_vehicle_is_there():
    h = Harness()
    h.frames(20)
    current = h.pipeline.current(h.now)
    assert current is not None and current.plate == "BK1234MA"
    assert current.event_id == h.events[0].event_id
    assert h.pipeline.current(h.now + 10.0) is None
    assert h.pipeline.last_plate().plate == "BK1234MA"


def test_crop_bottom_ratio_shortens_the_crop_sent_to_ocr():
    h = Harness(ocr=OcrConfig(crop_bottom_ratio=0.4, read_when="always"))
    h.frames(10)
    crop = h.reads[0][2]
    full_height = round((CENTRE_BOX[3] - CENTRE_BOX[1]) * 1.06)
    assert crop.shape[0] == pytest.approx(full_height * 0.6, abs=2)
