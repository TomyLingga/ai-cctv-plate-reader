"""The running system: readers, per-camera workers, shared models, storage and event bus."""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from alpr_id import __version__
from alpr_id.capture.reader import RtspReader
from alpr_id.capture.roi import roi_to_pixels
from alpr_id.config import AppConfig, CameraConfig, Roi, camera_url
from alpr_id.detect.onnx_detector import PlateDetector
from alpr_id.detect.providers import create_session, resolve_providers
from alpr_id.logging_setup import get_logger, mask_url
from alpr_id.ocr.recognizer import PlateOcr
from alpr_id.pipeline.camera_pipeline import CameraPipeline, VisiblePlate
from alpr_id.pipeline.events import EventBus, PlateEvent
from alpr_id.postprocess.correction import normalize_plate, settle_region
from alpr_id.postprocess.plate_format import REGION_NAMES, SPECIAL_CODES
from alpr_id.postprocess.regions import RegionBook
from alpr_id.postprocess.voting import OcrRead, PlateVoter
from alpr_id.storage.cameras import CameraStore
from alpr_id.storage.db import EventStore
from alpr_id.storage.media import MediaStore
from alpr_id.storage.regions import RegionStore
from alpr_id.timing import StageTimer, clock

log = get_logger(__name__)

ORIGIN_CONFIG = "config"  # listed in config/app.yaml, URL from the environment
ORIGIN_UI = "ui"  # added through the API, kept in the database
# All cameras share one detector and one OCR model. Past this many they would mostly skip frames.
MAX_CAMERAS = 16


class ManageError(Exception):
    """A camera or a region code could not be added, changed or removed.

    `code` is stable; clients switch on it.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class NotFoundError(ManageError):
    pass


class ConflictError(ManageError):
    pass


def load_models(config: AppConfig) -> tuple[PlateDetector, PlateOcr]:
    detector = PlateDetector(
        create_session(config.detector.model_path, config.runtime),
        conf_threshold=config.detector.conf_threshold,
        iou_threshold=config.detector.iou_threshold,
    )
    ocr = PlateOcr(config.ocr.model_path, config.ocr.plate_config_path, resolve_providers(config.runtime))
    return detector, ocr


@dataclass(frozen=True)
class CameraView:
    """Latest processed frame of a camera with what was found in it."""

    seq: int
    frame: np.ndarray
    visible: tuple[VisiblePlate, ...]


class CameraWorker(threading.Thread):
    """Takes the newest frame of one camera at the configured rate and runs the pipeline on it."""

    def __init__(
        self,
        camera: CameraConfig,
        reader: RtspReader,
        pipeline: CameraPipeline,
        timer: StageTimer,
        phase_s: float = 0.0,
        origin: str = ORIGIN_CONFIG,
        url: str = "",
    ) -> None:
        """`phase_s` delays the first frame so the cameras do not all ask for the models at once."""
        super().__init__(name=f"worker-{camera.id}", daemon=True)
        self.camera = camera
        self.reader = reader
        self.pipeline = pipeline
        self.timer = timer
        self.origin = origin
        self.url = url
        self.view: CameraView | None = None
        self._phase_s = phase_s
        self._stop_event = threading.Event()
        self._processed: deque[float] = deque(maxlen=200)

    def stop(self) -> None:
        self._stop_event.set()

    def processed_fps(self) -> float:
        now = clock()
        # Copy first: the worker thread appends while an API thread reads.
        return round(sum(1 for t in tuple(self._processed) if now - t <= 2.0) / 2.0, 1)

    def run(self) -> None:
        interval = 1.0 / self.camera.process_fps
        next_due = clock() + self._phase_s
        last_seq = -1
        while not self._stop_event.is_set():
            now = clock()
            if now < next_due:
                time.sleep(min(next_due - now, 0.005))
                continue
            # Slots are evenly spaced so the cameras take turns at the models. The newest frame
            # is taken as it is; it is at most one stream interval old.
            packet = self.reader.latest()
            if packet is None or packet.seq == last_seq:
                packet = self.reader.wait_newer(last_seq, timeout=0.2)
                if packet is None:
                    # No frames: tracks still have to time out and report what they have.
                    self.pipeline.expire(clock())
                    continue
            # When late, schedule from now: missed slots are skipped, not caught up.
            next_due = max(next_due + interval, clock())
            last_seq = packet.seq
            self.timer.add("frame_age", (clock() - packet.captured_at) * 1000.0)
            started = time.perf_counter()
            try:
                with self.timer.stage("convert"):
                    frame = packet.to_bgr()
                result = self.pipeline.process(frame, packet.captured_at, packet.wall_time)
            except Exception:
                log.exception("frame_failed", camera=self.camera.id)
                continue
            self.timer.add("total", (time.perf_counter() - started) * 1000.0)
            self.timer.add("capture_to_result", (clock() - packet.captured_at) * 1000.0)
            self._processed.append(clock())
            self.view = CameraView(packet.seq, frame, result.visible)


class Engine:
    def __init__(self, config: AppConfig, sources: dict[str, str] | None = None) -> None:
        """`sources` overrides the camera URLs, e.g. with recordings for tests.

        With `sources`, only the cameras of the config file run; cameras stored in the
        database are left alone.
        """
        self.config = config
        self.bus = EventBus()
        self._sources = sources
        self._started_at = time.time()
        self._inference_lock = threading.Lock() if config.runtime.serialize_inference else None
        self._workers: dict[str, CameraWorker] = {}
        # Guards the dict only and is never held while waiting on a thread: the async routes
        # read the dict from the event loop.
        self._cameras_lock = threading.Lock()
        # One add, change or removal at a time. Held for the whole operation.
        self._admin_lock = threading.Lock()
        self._launched = 0
        self._jpeg_cache: dict[tuple[str, int], tuple[int, bytes]] = {}
        self._jpeg_lock = threading.Lock()
        self._last_purge = 0.0
        self.detector: PlateDetector | None = None
        self.ocr: PlateOcr | None = None
        self.store: EventStore | None = None
        self.camera_store: CameraStore | None = None
        self.region_store: RegionStore | None = None
        self.media: MediaStore | None = None
        # One list of region codes for every camera. It starts as the built-in list and is
        # replaced by the stored one when the engine starts.
        self.regions = RegionBook(local_boost=config.postprocess.region_local_boost)

    # -- lifecycle ---------------------------------------------------------------------

    def start(self) -> None:
        self.detector, self.ocr = load_models(self.config)
        self.store = EventStore(self.config.storage.database)
        self.camera_store = CameraStore(self.config.storage.database)
        self.region_store = RegionStore(self.config.storage.database)
        self.regions.replace(self.region_store.codes())
        # Which codes pass this site most, from what was read before this start.
        for code, count in self.store.region_counts().items():
            self.regions.observe(code, count)
        self.media = MediaStore(self.config.storage.media_dir, self.config.timezone)
        self._purge_if_due()

        planned: list[tuple[CameraConfig, str, str]] = []
        for camera in self.config.cameras:
            url = self._sources[camera.id] if self._sources else camera_url(camera)
            planned.append((camera, url, ORIGIN_CONFIG))
        if self._sources is None:
            for stored in self.camera_store.all():
                # The config file wins: a stored camera that collides with it is not started.
                clash = self._clash(stored.camera, [camera for camera, _, _ in planned])
                if clash is not None:
                    log.warning("stored_camera_skipped", camera=stored.camera.id, reason=clash.message)
                    continue
                planned.append((stored.camera, stored.url, ORIGIN_UI))
        for index, (camera, url, origin) in enumerate(planned):
            self._launch(camera, url, origin, phase_s=index / len(planned) / camera.process_fps)
        log.info("engine_started", cameras=[camera.id for camera, _, _ in planned], providers=self.detector.providers)

    def stop(self) -> None:
        workers = self._snapshot()
        for worker in workers:
            worker.stop()
            worker.reader.stop()
        for worker in workers:
            worker.join(timeout=5)
            worker.reader.join(timeout=5)
        for worker in workers:
            if not worker.is_alive():
                worker.pipeline.flush()  # vehicles still being read get their event before the store closes
        if self.store is not None:
            self.store.close()
        if self.camera_store is not None:
            self.camera_store.close()
        if self.region_store is not None:
            self.region_store.close()
        log.info("engine_stopped")

    # -- cameras -----------------------------------------------------------------------

    def _snapshot(self) -> tuple[CameraWorker, ...]:
        with self._cameras_lock:
            return tuple(self._workers.values())

    def _worker(self, camera_id: str) -> CameraWorker:
        """KeyError for an unknown camera."""
        with self._cameras_lock:
            return self._workers[camera_id]

    def _worker_by_legacy(self, legacy_id: int) -> CameraWorker | None:
        return next((worker for worker in self._snapshot() if worker.camera.legacy_id == legacy_id), None)

    @staticmethod
    def _clash(camera: CameraConfig, others: list[CameraConfig]) -> ConflictError | None:
        for other in others:
            if other.id == camera.id:
                return ConflictError("duplicate_id", f"camera id {camera.id} is already in use")
            if camera.legacy_id is not None and other.legacy_id == camera.legacy_id:
                return ConflictError(
                    "duplicate_legacy_id", f"legacy number {camera.legacy_id} already belongs to camera {other.id}"
                )
        return None

    def _launch(self, camera: CameraConfig, url: str, origin: str, phase_s: float) -> CameraWorker:
        reader = RtspReader(camera.id, url, self.config.capture)
        timer = StageTimer()
        pipeline = CameraPipeline(
            camera, self.config, self.detector, self.ocr, timer,
            on_event=self._store_event,
            on_read=self._store_read if self.config.storage.save_reads else None,
            inference_lock=self._inference_lock,
            regions=self.regions,
        )
        worker = CameraWorker(camera, reader, pipeline, timer, phase_s=phase_s, origin=origin, url=url)
        with self._cameras_lock:
            self._workers[camera.id] = worker
            self._launched += 1
        self._forget_jpegs(camera.id)
        reader.start()
        worker.start()
        return worker

    def _retire(self, worker: CameraWorker) -> None:
        """Stop a worker that is no longer in the dict and let its vehicles report."""
        worker.stop()
        worker.reader.stop()
        worker.join(timeout=5)
        worker.reader.join(timeout=5)
        if not worker.is_alive():
            worker.pipeline.flush()

    def _forget_jpegs(self, camera_id: str) -> None:
        with self._jpeg_lock:
            for key in [key for key in self._jpeg_cache if key[0] == camera_id]:
                del self._jpeg_cache[key]

    def _next_phase(self, camera: CameraConfig) -> float:
        """Spread cameras added at run time over the frame interval."""
        return (self._launched % 8) / 8 / camera.process_fps

    def add_camera(self, camera: CameraConfig, url: str) -> dict[str, Any]:
        """Store a new camera and start reading it. Raises ConflictError."""
        with self._admin_lock:
            running = [worker.camera for worker in self._snapshot()]
            if len(running) >= MAX_CAMERAS:
                raise ConflictError("camera_limit", f"at most {MAX_CAMERAS} cameras can run on one service")
            clash = self._clash(camera, running)
            if clash is not None:
                raise clash
            self.camera_store.upsert(camera, url)
            worker = self._launch(camera, url, ORIGIN_UI, self._next_phase(camera))
            log.info("camera_added", camera=camera.id, url=mask_url(url), legacy_id=camera.legacy_id)
            return self._camera_payload(worker)

    def update_camera(self, camera: CameraConfig, url: str | None) -> dict[str, Any]:
        """Change a camera that was added through the API. `url=None` keeps the stored URL."""
        with self._admin_lock:
            old = self._require_editable(camera.id)
            clash = self._clash(camera, [w.camera for w in self._snapshot() if w.camera.id != camera.id])
            if clash is not None:
                raise clash
            new_url = url or old.url
            self.camera_store.upsert(camera, new_url)
            if new_url == old.url and camera.roi == old.camera.roi and camera.process_fps == old.camera.process_fps:
                # Only the description or the legacy number changed: the stream keeps running.
                old.camera = camera
                worker = old
            else:
                # The new worker takes the camera's place first, so its routes keep answering
                # while the old stream is being closed.
                worker = self._launch(camera, new_url, ORIGIN_UI, self._next_phase(camera))
                self._retire(old)
            log.info("camera_updated", camera=camera.id, url=mask_url(new_url), legacy_id=camera.legacy_id)
            return self._camera_payload(worker)

    def remove_camera(self, camera_id: str) -> None:
        """Stop and forget a camera that was added through the API. Its events stay."""
        with self._admin_lock:
            worker = self._require_editable(camera_id)
            with self._cameras_lock:
                del self._workers[camera_id]
            self.camera_store.delete(camera_id)
            self._retire(worker)
            self._forget_jpegs(camera_id)
            log.info("camera_removed", camera=camera_id)

    def _require_editable(self, camera_id: str) -> CameraWorker:
        try:
            worker = self._worker(camera_id)
        except KeyError:
            raise NotFoundError("unknown_camera", f"unknown camera {camera_id}") from None
        if worker.origin != ORIGIN_UI:
            raise ConflictError(
                "config_camera", f"camera {camera_id} is defined in the config file and can only be changed there"
            )
        return worker

    def stream_url(self, camera_id: str) -> str:
        """Stored URL of a camera, for a connection test. KeyError for an unknown camera."""
        return self._worker(camera_id).url

    def camera_sources(self) -> list[dict[str, Any]]:
        """Where each camera comes from, with the password hidden. For administrators."""
        return [
            {
                "id": worker.camera.id,
                "origin": worker.origin,
                "url": mask_url(worker.url),
                "url_env": worker.camera.url_env or None,
            }
            for worker in self._snapshot()
        ]

    # -- region codes ------------------------------------------------------------------

    def region_list(self) -> list[dict[str, Any]]:
        """Every listed code with its area and how often it was read here."""
        seen = self.regions.seen()
        return [{**row, "seen": seen.get(row["code"], 0)} for row in self.region_store.all()]

    def add_region(self, code: str, name: str) -> dict[str, Any]:
        with self._admin_lock:
            if code in SPECIAL_CODES:
                raise ConflictError("special_code", f"{code} is a diplomatic or state code and is always accepted")
            if not self.region_store.add(code, name):
                raise ConflictError("duplicate_code", f"region code {code} is already listed")
            self.regions.replace(self.region_store.codes())
            log.info("region_added", code=code, name=name)
            return {"code": code, "name": name, "builtin": code in REGION_NAMES, "seen": self.regions.seen().get(code, 0)}

    def remove_region(self, code: str) -> None:
        with self._admin_lock:
            if not self.region_store.remove(code):
                raise NotFoundError("unknown_code", f"region code {code} is not listed")
            self.regions.replace(self.region_store.codes())
            log.info("region_removed", code=code)

    def restore_regions(self) -> int:
        """Put back the built-in codes that were removed. Returns how many came back."""
        with self._admin_lock:
            added = self.region_store.restore_defaults()
            self.regions.replace(self.region_store.codes())
            log.info("regions_restored", added=added)
            return added

    # -- sinks called from camera threads ----------------------------------------------

    def _purge_if_due(self) -> None:
        """Drop media older than the retention period, at most once a day."""
        now = time.time()
        with self._jpeg_lock:
            if now - self._last_purge < 24 * 3600:
                return
            self._last_purge = now
        removed = self.media.purge_older_than(self.config.storage.retention_days)
        if removed:
            log.info("media_purged", day_folders=removed)

    def _store_event(self, event: PlateEvent, crop: np.ndarray | None, frame: np.ndarray | None) -> PlateEvent:
        self._purge_if_due()
        short = event.event_id[:8]
        crop_url = self.media.save(crop, event.camera_id, event.ts_epoch, f"{short}_crop") if crop is not None else None
        snapshot_url = None
        if frame is not None and self.config.storage.save_snapshot:
            snapshot_url = self.media.save(frame, event.camera_id, event.ts_epoch, f"{short}_frame", quality=85)
        event = event.with_media(crop_url, snapshot_url)
        self.store.insert(event)
        self.bus.publish(event.to_dict())
        log.info(
            "plate_event", event_id=event.event_id, camera=event.camera_id, plate=event.plate,
            confidence=round(event.confidence, 3), format_valid=event.format_valid, reads=event.reads,
            track_id=event.track_id, alternatives=list(event.alternatives),
        )
        return event

    def _store_read(self, camera_id: str, track_id: int, crop: np.ndarray, read: OcrRead, wall_time: float) -> None:
        name = f"t{track_id:05d}_{read.text or 'EMPTY'}_{int(read.confidence * 100):03d}_{uuid.uuid4().hex[:6]}"
        self.media.save(crop, f"{camera_id}/reads", wall_time, name, quality=95)

    # -- queries from the API ----------------------------------------------------------

    def health(self) -> dict[str, Any]:
        statuses = [worker.reader.status() for worker in self._snapshot()]
        return {
            "status": "ok",
            "version": __version__,
            "uptime_s": round(time.time() - self._started_at, 1),
            "cameras_total": len(statuses),
            "cameras_connected": sum(1 for s in statuses if s.connected),
            "providers": self.detector.providers if self.detector else [],
            "events_stored": self.store.count() if self.store else 0,
        }

    def cameras(self) -> list[dict[str, Any]]:
        return [self._camera_payload(worker) for worker in self._snapshot()]

    def _camera_payload(self, worker: CameraWorker) -> dict[str, Any]:
        camera = worker.camera
        status = worker.reader.status()
        latency = worker.timer.summary()
        return {
            "id": camera.id,
            "legacy_id": camera.legacy_id,
            "description": camera.description,
            "origin": worker.origin,
            "roi": list(camera.roi),
            "focus": list(camera.focus) if camera.focus is not None else None,
            "process_fps": camera.process_fps,
            "connected": status.connected,
            "fps": status.fps,
            "processed_fps": worker.processed_fps(),
            "last_frame_age_s": status.last_frame_age_s,
            "lag_s": status.lag_s,
            "resolution": [status.width, status.height] if status.width else None,
            "codec": status.codec,
            "reconnects": status.reconnects,
            "last_error": status.last_error,
            "latency_ms": {k: latency[k] for k in ("total", "frame_age", "capture_to_result") if k in latency},
            "current": self._plate_payload(camera.id, worker.pipeline.current(clock())),
            # Plates being followed, read or not: a truck still rolling shows up here first.
            "plates_in_view": worker.pipeline.in_view(clock()),
        }

    def has_camera(self, camera_id: str) -> bool:
        with self._cameras_lock:
            return camera_id in self._workers

    def current(self, camera_id: str) -> dict[str, Any]:
        """KeyError for an unknown camera."""
        worker = self._worker(camera_id)
        plate = worker.pipeline.current(clock())
        return self._plate_payload(camera_id, plate)

    @staticmethod
    def _plate_payload(camera_id: str, plate: VisiblePlate | None) -> dict[str, Any]:
        if plate is None:
            return {"camera_id": camera_id, "visible": False, "plate": None}
        return {
            "camera_id": camera_id,
            "visible": True,
            "plate": plate.plate,
            "plate_formatted": plate.plate_formatted,
            "confidence": None if plate.confidence is None else round(plate.confidence, 4),
            "format_valid": plate.format_valid,
            "region_known": plate.region_known,
            "region_corrected": plate.region_corrected,
            "raw": plate.raw,
            "alternatives": list(plate.alternatives),
            "track_id": plate.track_id,
            "bbox": list(plate.box),
            "reads": plate.reads,
            "event_id": plate.event_id,
        }

    def legacy_plate(self, legacy_id: int) -> dict[str, Any] | None:
        """Answer in the shape of the old service: plate, raw, fps, status."""
        worker = self._worker_by_legacy(legacy_id)
        if worker is None:
            return None
        status = worker.reader.status()
        now = clock()
        plate = worker.pipeline.current(now)
        if plate is None:
            # The old service kept a plate for a few seconds after it was last read.
            last = worker.pipeline.last_plate()
            if last is not None and now - last.last_seen <= self.config.api.legacy_hold_s:
                plate = last
        if plate is None or not plate.plate:
            text = ""
        elif self.config.api.legacy_plate_format == "spaced":
            text = plate.plate_formatted or ""
        else:
            text = plate.plate
        state = "offline" if not status.connected else ("lagging" if status.fps < 1.0 else "online")
        return {"plate": text, "raw": (plate.raw or "") if plate else "", "fps": float(status.fps), "status": state}

    def camera_id_for_legacy(self, legacy_id: int) -> str | None:
        worker = self._worker_by_legacy(legacy_id)
        return worker.camera.id if worker else None

    def video_jpeg(self, camera_id: str, width: int | None = None, crop: Roi | None = None) -> bytes:
        """Latest processed frame as JPEG, scaled down, with boxes and plate text drawn.

        `crop` is a part of the frame as fractions [x1, y1, x2, y2]: that part is cut from the
        full-resolution frame before scaling, so an enlarged view shows real detail.
        KeyError for an unknown camera.
        """
        worker = self._worker(camera_id)
        width = width or self.config.api.video_width
        view = worker.view
        fresh = view is not None and worker.reader.status().connected
        if crop is not None:
            # Not cached: every viewer picks its own part of the frame.
            image = self._render(view, width, crop) if fresh else self._no_signal(width)
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return encoded.tobytes() if ok else b""
        seq = view.seq if fresh else -1
        key = (camera_id, width)
        with self._jpeg_lock:
            cached = self._jpeg_cache.get(key)
            if cached is not None and cached[0] == seq:
                return cached[1]
        image = self._render(view, width) if fresh else self._no_signal(width)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        data = encoded.tobytes() if ok else b""
        with self._jpeg_lock:
            self._jpeg_cache[key] = (seq, data)
        return data

    @staticmethod
    def _render(view: CameraView, target_width: int, crop: Roi | None = None) -> np.ndarray:
        frame, left, top = view.frame, 0, 0
        if crop is not None:
            part = roi_to_pixels(crop, frame.shape[1], frame.shape[0])
            frame, left, top = part.crop(frame), part.x1, part.y1
        height, width = frame.shape[:2]
        scale = min(target_width / width, 1.0)  # never enlarged here: the browser does that
        size = (max(int(width * scale), 1), max(int(height * scale), 1))
        # Without scaling the crop is still a view into the frame, and drawing needs its own copy.
        image = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if scale < 1.0 else frame.copy()
        now = clock()
        thickness = 2 if image.shape[1] >= 960 else 1
        for plate in view.visible:
            if now - plate.last_seen > 1.5:
                continue
            x1, y1, x2, y2 = (int((v - offset) * scale) for v, offset in zip(plate.box, (left, top, left, top), strict=True))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            if plate.plate_formatted:
                cv2.putText(image, plate.plate_formatted, (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7 if thickness == 2 else 0.5, (0, 255, 255), thickness, cv2.LINE_AA)
        return image

    @staticmethod
    def _no_signal(width: int) -> np.ndarray:
        height = width * 9 // 16
        image = np.zeros((height, width, 3), dtype=np.uint8)
        scale = width / 800
        (text_w, text_h), _ = cv2.getTextSize("NO SIGNAL", cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.putText(image, "NO SIGNAL", ((width - text_w) // 2, (height + text_h) // 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 255), 2, cv2.LINE_AA)
        return image

    def recognize(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Detect and read plates in a single still image. No tracking or voting."""
        lock = self._inference_lock
        if lock is not None:
            lock.acquire()
        try:
            detections = self.detector.detect(image)
            crops, kept = [], []
            for det in detections:
                x1, y1, x2, y2 = (int(round(v)) for v in det.box)
                crop = image[max(y1, 0) : y2, max(x1, 0) : x2]
                if crop.size:
                    crops.append(crop)
                    kept.append(det)
            reads = self.ocr.read(crops)
        finally:
            if lock is not None:
                lock.release()
        results = []
        post = self.config.postprocess
        for det, read in zip(kept, reads, strict=True):
            plate, confidence = normalize_plate(read.text, self.regions), read.confidence
            if post.fix_unknown_region:
                # One read is a vote of one: the same rule as for a tracked vehicle applies.
                voter = PlateVoter(self.ocr.alphabet, self.ocr.pad_char)
                voter.add(read)
                vote = voter.result()
                if vote is not None:
                    plate, vote = settle_region(plate, vote, self.ocr.alphabet, self.regions)
                    confidence = vote.confidence
            results.append(
                {
                    "plate": plate.plate,
                    "plate_formatted": plate.formatted,
                    "confidence": round(confidence, 4),
                    "format_valid": plate.format_valid,
                    "region_known": plate.region_known,
                    "region_corrected": plate.region_corrected,
                    "detection_confidence": round(det.confidence, 4),
                    "bbox": [int(round(v)) for v in det.box],
                    "raw": read.text,
                }
            )
        return results
