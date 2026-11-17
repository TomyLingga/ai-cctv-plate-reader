"""Everything that happens to one frame of one camera, and the state kept between frames.

detect -> track -> pick the best crop per track -> OCR (a few times per track) -> vote ->
emit one event per vehicle.
"""
from __future__ import annotations

import math
import threading
import uuid
from collections import deque
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np

from alpr_id.capture.roi import PixelRect, clamp_box, roi_to_pixels
from alpr_id.config import AppConfig, CameraConfig
from alpr_id.detect.onnx_detector import Detection, PlateDetector
from alpr_id.logging_setup import get_logger
from alpr_id.ocr.quality import crop_quality
from alpr_id.ocr.recognizer import PlateOcr
from alpr_id.pipeline.events import PlateEvent, iso_timestamp
from alpr_id.postprocess.correction import PlateResult, normalize_plate, settle_region
from alpr_id.postprocess.regions import RegionBook
from alpr_id.postprocess.voting import OcrRead, PlateVoter, VoteResult
from alpr_id.timing import StageTimer
from alpr_id.track.bytetrack import CameraTracker, TrackedPlate, box_iou

log = get_logger(__name__)

EDGE_MARGIN_PX = 3
ACTIVE_WINDOW_S = 1.5
ADOPT_MIN_IOU = 0.3
# The focus pass costs a second detector call. With nothing in view it runs on every
# FOCUS_IDLE_EVERY-th frame only (twice a second at 8 fps); once either pass finds a plate it
# runs on every frame until FOCUS_ACTIVE_S after the last detection. On the GTX 1060 one
# call takes about 15 ms, and two calls per frame for four cameras at 8 fps would fill it.
FOCUS_IDLE_EVERY = 4
FOCUS_ACTIVE_S = 3.0
DROPPED = ""  # event_id of a track whose result was too weak to report

EventSink = Callable[[PlateEvent, np.ndarray | None, np.ndarray | None], PlateEvent]
ReadSink = Callable[[str, int, np.ndarray, OcrRead, float], None]


def merge_detections(wide: list[Detection], focus: list[Detection], iou_threshold: float) -> list[Detection]:
    """Union of the two detection passes over one frame, in frame coordinates.

    Where both passes found the same plate (boxes overlapping by `iou_threshold` or more),
    the more confident box is kept, so a plate never comes out twice.
    """
    merged = list(wide)
    for extra in focus:
        for index, kept in enumerate(merged):
            if box_iou(kept.box, extra.box) >= iou_threshold:
                if extra.confidence > kept.confidence:
                    merged[index] = extra
                break
        else:
            merged.append(extra)
    return merged


@dataclass
class Candidate:
    """A crop that may be sent to OCR, with the frame it came from."""

    score: float
    crop: np.ndarray
    box: tuple[int, int, int, int]
    frame: np.ndarray | None  # full frame, released once the event is stored
    wall_time: float


@dataclass
class TrackState:
    track_id: int
    first_seen: float
    last_seen: float
    box: tuple[float, float, float, float]
    voter: PlateVoter
    reads: int = 0
    last_ocr_at: float = -math.inf
    read_score: float = 0.0  # quality of the best crop read so far
    pending: Candidate | None = None  # best crop since the last OCR pass
    best: Candidate | None = None  # best crop whose read produced text
    vote: VoteResult | None = None
    plate: PlateResult | None = None
    event_id: str | None = None  # None while reading, DROPPED, or the id of the event
    retry_at: float = math.inf  # when a dropped vehicle is read afresh
    # Recent positions of the plate: (time, centre x, centre y, width).
    path: deque[tuple[float, float, float, float]] = field(default_factory=deque)
    still: bool = False  # the plate has not moved for a moment
    was_still: bool = False  # it stood still at some point since it appeared
    read_still: bool = False  # at least one of its reads was taken while it stood
    # It appeared right after a standing vehicle was seen: the same truck under a new id.
    shadow: bool = False
    release_at: float = math.inf  # when a held passing read may become an event


@dataclass(frozen=True)
class VisiblePlate:
    """What is known about one track right now. Safe to hand to other threads."""

    track_id: int
    box: tuple[int, int, int, int]
    plate: str | None
    plate_formatted: str | None
    raw: str | None
    confidence: float | None
    format_valid: bool | None
    region_known: bool | None
    alternatives: tuple[str, ...]
    reads: int
    event_id: str | None
    last_seen: float
    region_corrected: bool = False


@dataclass(frozen=True)
class FrameResult:
    visible: tuple[VisiblePlate, ...]
    events: tuple[PlateEvent, ...] = field(default_factory=tuple)


class CameraPipeline:
    def __init__(
        self,
        camera: CameraConfig,
        config: AppConfig,
        detector: PlateDetector,
        ocr: PlateOcr,
        timer: StageTimer,
        on_event: EventSink | None = None,
        on_read: ReadSink | None = None,
        inference_lock: threading.Lock | None = None,
        regions: RegionBook | None = None,
    ) -> None:
        """`regions` is the list of region codes shared by all cameras; a pipeline on its own
        gets the built-in list."""
        self.camera = camera
        self._config = config
        self._detector = detector
        self._ocr = ocr
        self._timer = timer
        self._on_event = on_event
        self._on_read = on_read
        self._lock = inference_lock
        self._regions = regions if regions is not None else RegionBook(local_boost=config.postprocess.region_local_boost)
        self._tracker = CameraTracker(config.tracking, frame_rate=camera.process_fps)
        self._tracks: dict[int, TrackState] = {}
        self._recent_events: dict[str, tuple[str, float]] = {}
        self._roi_cache: tuple[tuple[int, int], PixelRect, PixelRect | None] | None = None
        self._frames = 0
        self._focus_active_until = -math.inf
        self._visible: tuple[VisiblePlate, ...] = ()
        self._last_plate: VisiblePlate | None = None
        self._standing_seen_at = -math.inf  # last time a vehicle that stood here was in view
        self._held: list[TrackState] = []  # vehicles that passed, waiting for their one read

    # -- read side, callable from any thread -------------------------------------------

    def visible(self) -> tuple[VisiblePlate, ...]:
        return self._visible

    def _reportable(self, plate: VisiblePlate, now: float) -> bool:
        """Seen moments ago and read well enough to show to a client."""
        return bool(
            plate.plate
            and now - plate.last_seen <= ACTIVE_WINDOW_S
            and (plate.confidence or 0.0) >= self._config.voting.min_event_confidence
        )

    def current(self, now: float) -> VisiblePlate | None:
        """Best plate among the tracks seen within the last moments, or None."""
        fresh = [v for v in self._visible if self._reportable(v, now)]
        return max(fresh, key=lambda v: v.confidence or 0.0) if fresh else None

    def last_plate(self) -> VisiblePlate | None:
        """Most recent plate that was readable, even if its vehicle has left."""
        return self._last_plate

    def in_view(self, now: float) -> int:
        """How many plates are being followed right now, read or not."""
        return sum(1 for plate in self._visible if now - plate.last_seen <= ACTIVE_WINDOW_S)

    # -- frame processing, one thread per camera ---------------------------------------

    def _areas(self, width: int, height: int) -> tuple[PixelRect, PixelRect | None]:
        """The search area and, if the camera has one, the focus area, in pixels of this frame size."""
        if self._roi_cache is None or self._roi_cache[0] != (width, height):
            focus = self.camera.focus
            self._roi_cache = (
                (width, height),
                roi_to_pixels(self.camera.roi, width, height),
                roi_to_pixels(focus, width, height) if focus is not None else None,
            )
        return self._roi_cache[1], self._roi_cache[2]

    def _inference(self):
        return self._lock if self._lock is not None else nullcontext()

    def _in_focus(self, output, box, focus: PixelRect) -> list[Detection]:
        """Detections of the focus pass in frame coordinates, without plates cut by its border.

        A plate that crosses the border of the focus area gets a box around the visible part
        only. The wide pass sees that plate whole, so the partial box is not needed, and OCR
        would read it confidently and wrongly.
        """
        kept = []
        for found in self._detector.postprocess(output, box, (focus.height, focus.width)):
            if (
                found.x1 <= EDGE_MARGIN_PX
                or found.y1 <= EDGE_MARGIN_PX
                or found.x2 >= focus.width - EDGE_MARGIN_PX
                or found.y2 >= focus.height - EDGE_MARGIN_PX
            ):
                continue
            kept.append(found.shifted(focus.x1, focus.y1))
        return kept

    def process(self, frame: np.ndarray, captured_at: float, wall_time: float) -> FrameResult:
        height, width = frame.shape[:2]
        roi, focus = self._areas(width, height)
        self._frames += 1
        if focus is not None and captured_at > self._focus_active_until and self._frames % FOCUS_IDLE_EVERY:
            focus = None  # nothing in view: the focus pass only looks in now and then

        # Two views of the same frame: the whole search area shrunk to the detector's input,
        # and the spot where vehicles stand at (close to) full resolution. Each finds plates
        # the other misses, so their detections are merged before tracking.
        with self._timer.stage("preprocess"):
            tensor, box = self._detector.preprocess(roi.crop(frame))
            if focus is not None:
                focus_tensor, focus_box = self._detector.preprocess(focus.crop(frame))
        with self._inference(), self._timer.stage("detect"):
            output = self._detector.infer(tensor)
            if focus is not None:
                focus_output = self._detector.infer(focus_tensor)
        with self._timer.stage("track"):
            detections = [d.shifted(roi.x1, roi.y1) for d in self._detector.postprocess(output, box, (roi.height, roi.width))]
            if focus is not None:
                detections = merge_detections(
                    detections, self._in_focus(focus_output, focus_box, focus), self._config.detector.iou_threshold
                )
            if detections:
                self._focus_active_until = captured_at + FOCUS_ACTIVE_S
            tracked = self._tracker.update(detections, captured_at)

        with self._timer.stage("select"):
            for plate in tracked:
                self._observe(plate, frame, roi, captured_at, wall_time)

        self._run_ocr(captured_at)

        events: list[PlateEvent] = []
        with self._timer.stage("emit"):
            exhausted_at = self._config.ocr.max_reads_per_track
            for state in self._tracks.values():
                if state.event_id is not None:
                    continue
                exhausted = state.reads >= exhausted_at
                if state.vote is None:
                    if exhausted:
                        self._drop(state, captured_at)
                elif exhausted or self._stable(state):
                    self._emit(state, events, captured_at)
            self._close_stale(captured_at, events)
            self._publish_view(captured_at)
        return FrameResult(self._visible, tuple(events))

    def expire(self, now: float) -> tuple[PlateEvent, ...]:
        """Close tracks not seen for too long. For when the camera delivers no frames."""
        events: list[PlateEvent] = []
        self._close_stale(now, events)
        self._publish_view(now)
        return tuple(events)

    def flush(self) -> tuple[PlateEvent, ...]:
        """Close every track, as at the end of a recording or when the service stops."""
        return self.expire(math.inf)

    def _new_voter(self) -> PlateVoter:
        post = self._config.postprocess
        return PlateVoter(self._ocr.alphabet, self._ocr.pad_char, post.confusable_letters, post.ambiguity_margin)

    def _observe(self, plate: TrackedPlate, frame: np.ndarray, roi: PixelRect, captured_at: float, wall_time: float) -> None:
        state = self._tracks.get(plate.track_id)
        if state is None:
            state = self._adopt(plate, captured_at)
        if state is None:
            state = TrackState(plate.track_id, captured_at, captured_at, plate.box, self._new_voter())
            state.shadow = captured_at - self._standing_seen_at <= self._config.ocr.passing_quiet_s
            self._tracks[plate.track_id] = state
        state.last_seen = captured_at
        state.box = plate.box
        self._note_motion(state, captured_at)

        if state.event_id:
            # Still here: keep the event fresh so a new track of this vehicle is not a new event.
            if state.plate is not None:
                self._recent_events[state.plate.plate] = (state.event_id, wall_time)
            return
        if state.event_id == DROPPED:
            if captured_at < state.retry_at:
                return
            # The first attempt gave nothing usable. The vehicle is still here, so start over.
            state.voter = self._new_voter()
            state.reads, state.read_score, state.last_ocr_at = 0, 0.0, -math.inf
            state.pending = state.best = state.vote = state.plate = None
            state.event_id, state.retry_at = None, math.inf

        ocr = self._config.ocr
        if state.reads >= ocr.max_reads_per_track:
            return
        if plate.width < ocr.min_plate_width_px:
            return
        x1, y1, x2, y2 = plate.box
        # A box on the border of the search area is a plate cut in half. OCR reads such a
        # crop confidently and wrongly, so it is never sent.
        if (
            x1 <= roi.x1 + EDGE_MARGIN_PX
            or y1 <= roi.y1 + EDGE_MARGIN_PX
            or x2 >= roi.x2 - EDGE_MARGIN_PX
            or y2 >= roi.y2 - EDGE_MARGIN_PX
        ):
            return
        rect = clamp_box(plate.box, frame.shape[1], frame.shape[0], ocr.crop_padding)
        if rect is None:
            return
        crop = rect.crop(frame)
        if ocr.crop_bottom_ratio > 0:
            crop = crop[: max(int(round(crop.shape[0] * (1.0 - ocr.crop_bottom_ratio))), 2)]
        score = crop_quality(crop)
        if state.pending is None or score > state.pending.score:
            state.pending = Candidate(score, crop.copy(), (rect.x1, rect.y1, rect.x2, rect.y2), frame, wall_time)

    def _note_motion(self, state: TrackState, captured_at: float) -> None:
        """Record where the plate is and decide whether it is standing still.

        Still means: over the last `still_window_s` the centre of the box stayed within
        `still_tolerance` of the plate's width, and the width itself hardly changed. The
        detector's boxes jitter by a few pixels on a parked vehicle, far less than that.
        """
        ocr = self._config.ocr
        x1, y1, x2, y2 = state.box
        width = x2 - x1
        centre_x, centre_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        path = state.path
        path.append((captured_at, centre_x, centre_y, width))
        while captured_at - path[0][0] > ocr.still_window_s * 2.0:
            path.popleft()

        still = False
        # Not enough history yet means unknown, which counts as moving.
        if width > 0 and captured_at - path[0][0] >= ocr.still_window_s * 0.9:
            reach = ocr.still_tolerance * width
            still = all(
                math.hypot(x - centre_x, y - centre_y) <= reach and abs(w - width) <= 2.0 * reach
                for t, x, y, w in path
                if captured_at - t <= ocr.still_window_s
            )
        if still and not state.still and state.event_id is None and ocr.read_when != "always":
            # The crop kept so far was taken while the vehicle was rolling. Reads start from
            # the standing plate.
            state.pending = None
        state.still = still
        state.was_still = state.was_still or still
        if state.was_still:
            self._standing_seen_at = captured_at

    def _adopt(self, plate: TrackedPlate, captured_at: float) -> TrackState | None:
        """Hand the state of a track that just went missing to the new track at the same spot.

        The tracker gives a new id to a plate it lost for a few seconds, e.g. while someone
        stood in front of a parked truck. Without this the vehicle would be read again from
        scratch and could produce a second, different event.
        """
        best: TrackState | None = None
        best_iou = ADOPT_MIN_IOU
        for state in self._tracks.values():
            if state.last_seen >= captured_at:  # still visible in this frame, so a different plate
                continue
            overlap = box_iou(state.box, plate.box)
            if overlap >= best_iou:
                best, best_iou = state, overlap
        if best is None:
            return None
        del self._tracks[best.track_id]
        best.track_id = plate.track_id
        self._tracks[plate.track_id] = best
        return best

    def _read_due(self, state: TrackState, captured_at: float) -> bool:
        """Whether this track's waiting crop should be read now.

        The first reads come quickly so a result exists soon. After that the remaining reads
        are kept for crops clearly better than anything read so far, typically as the vehicle
        comes closer. If nothing better turns up for a while, the waiting crop is read anyway.
        """
        ocr = self._config.ocr
        if state.pending is None or state.event_id is not None or state.reads >= ocr.max_reads_per_track:
            return False
        if ocr.read_when != "always" and not state.still:
            return False  # a rolling plate is not read while it is in view
        waited = captured_at - state.last_ocr_at
        if waited < ocr.read_interval_s:
            return False
        if state.reads < self._config.voting.min_reads:
            return True
        return state.pending.score > state.read_score * (1.0 + ocr.read_improvement) or waited >= ocr.stale_read_s

    def _run_ocr(self, captured_at: float) -> None:
        self._read([state for state in self._tracks.values() if self._read_due(state, captured_at)], captured_at)

    def _settle(self, vote: VoteResult) -> tuple[PlateResult | None, VoteResult]:
        """Plate for a vote, with an unlisted region code replaced when that is switched on.

        None when the text cannot be a plate at all.
        """
        plate = normalize_plate(vote.text, self._regions)
        # Every plate has a number. Text without a single digit is something else the
        # detector took for a plate, such as the lettering on a guard's vest.
        if not any(char.isdigit() for char in plate.plate):
            return None, vote
        if self._config.postprocess.fix_unknown_region:
            return settle_region(plate, vote, self._ocr.alphabet, self._regions)
        return plate, vote

    def _read(self, due: list[TrackState], captured_at: float) -> None:
        """Run OCR on the waiting crop of every given track and update its vote."""
        if not due:
            return
        with self._inference(), self._timer.stage("ocr"):
            reads = self._ocr.read([state.pending.crop for state in due])
        with self._timer.stage("vote"):
            for state, read in zip(due, reads, strict=True):
                candidate = state.pending
                state.pending = None
                state.reads += 1
                state.read_still = state.read_still or state.still
                state.last_ocr_at = captured_at
                state.read_score = max(state.read_score, candidate.score)
                state.voter.add(read)
                if read.text and (state.best is None or candidate.score > state.best.score):
                    state.best = candidate
                vote = state.voter.result()
                state.plate, state.vote = self._settle(vote) if vote else (None, None)
                if self._on_read is not None:
                    try:
                        self._on_read(self.camera.id, state.track_id, candidate.crop, read, candidate.wall_time)
                    except Exception:
                        log.exception("read_sink_failed", camera=self.camera.id)

    def _stable(self, state: TrackState) -> bool:
        """Enough agreeing reads to report before the read budget is used up."""
        voting = self._config.voting
        vote = state.vote
        return (
            vote.reads_used >= voting.min_reads
            and vote.confidence >= voting.emit_confidence
            # The average hides a single character that is still a coin toss.
            and min(vote.char_confidences) >= voting.min_char_confidence
            and state.plate is not None
            and state.plate.format_valid
        )

    def _drop(self, state: TrackState, captured_at: float) -> None:
        state.event_id = DROPPED
        pause = self._config.ocr.retry_after_s
        state.retry_at = captured_at + pause if pause > 0 else math.inf
        log.info(
            "track_dropped", camera=self.camera.id, track_id=state.track_id, reads=state.reads,
            confidence=None if state.vote is None else round(state.vote.confidence, 3),
        )

    def _emit(self, state: TrackState, events: list[PlateEvent], captured_at: float) -> None:
        vote, plate, best = state.vote, state.plate, state.best
        voting = self._config.voting
        if vote is None or plate is None or best is None or vote.confidence < voting.min_event_confidence:
            self._drop(state, captured_at)
            return
        passing = self._config.ocr.read_when == "stopped_or_passing" and not state.read_still
        if passing and not (plate.format_valid and vote.confidence >= voting.min_passing_confidence):
            self._drop(state, captured_at)
            return

        window = self._config.voting.dedup_window_s
        self._recent_events = {k: v for k, v in self._recent_events.items() if best.wall_time - v[1] <= window}
        seen = self._recent_events.get(plate.plate)
        if seen is not None:
            # Same vehicle under a new track id, e.g. after someone walked in front of it.
            state.event_id = seen[0]
            self._recent_events[plate.plate] = (seen[0], best.wall_time)
            return

        event = PlateEvent(
            event_id=uuid.uuid4().hex,
            camera_id=self.camera.id,
            plate=plate.plate,
            plate_formatted=plate.formatted,
            confidence=vote.confidence,
            format_valid=plate.format_valid,
            timestamp=iso_timestamp(best.wall_time, self._config.timezone),
            track_id=state.track_id,
            bbox=best.box,
            region_known=plate.region_known,
            alternatives=self._alternatives(vote, plate),
            reads=state.reads,
            raw=vote.text,
            ts_epoch=best.wall_time,
            region_corrected=plate.region_corrected,
            stopped=state.read_still,
        )
        # The event counts as emitted even if storing it fails. Otherwise a full disk or a
        # locked database would make every following frame try again and write new files.
        state.event_id = event.event_id
        self._recent_events[plate.plate] = (event.event_id, best.wall_time)
        # Only codes that were read as they are count as "seen here"; a replaced code would
        # otherwise vote for itself the next time.
        if plate.parts is not None and plate.format_valid and not plate.region_corrected and plate.parts.prefix in self._regions:
            self._regions.observe(plate.parts.prefix)
        if self._on_event is not None:
            try:
                event = self._on_event(event, best.crop, best.frame)
            except Exception:
                log.exception("event_sink_failed", camera=self.camera.id, event_id=event.event_id, plate=event.plate)
        # The frame is no longer needed once the event is stored; keep the small crop only.
        state.best = Candidate(best.score, best.crop, best.box, None, best.wall_time)
        events.append(event)

    def _alternatives(self, vote: VoteResult, plate: PlateResult) -> tuple[str, ...]:
        others = (normalize_plate(text, self._regions).plate for text in vote.alternatives)
        return tuple(dict.fromkeys(other for other in others if other != plate.plate))

    def _close_stale(self, captured_at: float, events: list[PlateEvent]) -> None:
        timeout = self._config.tracking.track_timeout_s
        ocr = self._config.ocr
        for track_id in [tid for tid, s in self._tracks.items() if captured_at - s.last_seen > timeout]:
            state = self._tracks.pop(track_id)
            if state.event_id is None and state.vote is not None:
                self._emit(state, events, captured_at)
            elif (
                ocr.read_when == "stopped_or_passing"
                and state.event_id is None
                and state.reads == 0
                and state.pending is not None
                and not state.was_still
                and not state.shadow
            ):
                # It left without ever standing still. Its best frame is kept for a while:
                # if the same truck turns out to stand here, that read is the one that counts.
                state.release_at = captured_at + ocr.passing_quiet_s
                self._held.append(state)
        self._release_held(captured_at, events)

    def _release_held(self, captured_at: float, events: list[PlateEvent]) -> None:
        """Read the vehicles that only passed, once it is clear none of them stopped here."""
        if not self._held:
            return
        due: list[TrackState] = []
        waiting: list[TrackState] = []
        for state in self._held:
            if self._standing_seen_at > state.last_seen:
                continue  # a vehicle stood here after it passed: the same one, read properly
            (due if captured_at >= state.release_at else waiting).append(state)
        self._held = waiting
        if not due:
            return
        try:
            self._read(due, captured_at)
        except Exception:
            log.exception("passing_read_failed", camera=self.camera.id)
            return
        for state in due:
            if state.vote is not None:
                self._emit(state, events, captured_at)

    def _publish_view(self, captured_at: float) -> None:
        visible = []
        for state in self._tracks.values():
            plate, vote = state.plate, state.vote
            alternatives = self._alternatives(vote, plate) if plate is not None and vote is not None else ()
            visible.append(
                VisiblePlate(
                    track_id=state.track_id,
                    box=tuple(int(round(v)) for v in state.box),
                    plate=plate.plate if plate else None,
                    plate_formatted=plate.formatted if plate else None,
                    raw=vote.text if vote else None,
                    confidence=vote.confidence if vote else None,
                    format_valid=plate.format_valid if plate else None,
                    region_known=plate.region_known if plate else None,
                    alternatives=alternatives,
                    reads=state.reads,
                    event_id=state.event_id or None,
                    last_seen=state.last_seen,
                    region_corrected=plate.region_corrected if plate else False,
                )
            )
        self._visible = tuple(visible)
        fresh = [v for v in visible if self._reportable(v, captured_at)]
        if fresh:
            self._last_plate = max(fresh, key=lambda v: v.confidence or 0.0)
