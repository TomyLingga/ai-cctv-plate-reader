"""RTSP reader: one thread per camera that always holds the most recent frame.

Every frame has to be decoded because video frames depend on the ones before them, but only
the latest decoded frame is kept. A consumer that is slower than the camera therefore skips
frames instead of falling behind. Conversion to a BGR array is left to the consumer, so
skipped frames cost no colour conversion.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import av
import numpy as np

from alpr_id.config import CaptureConfig
from alpr_id.logging_setup import get_logger, mask_url
from alpr_id.timing import clock

log = get_logger(__name__)

MAX_BAD_PACKETS = 30
LAG_WINDOW_S = 60


class _FellBehindError(Exception):
    """The reader is decoding frames that are too old; reconnect to drop the backlog."""


@dataclass(frozen=True)
class FramePacket:
    frame: av.VideoFrame
    seq: int
    captured_at: float  # timing.clock() when the frame was decoded
    wall_time: float  # time.time() at the same moment

    def to_bgr(self) -> np.ndarray:
        return self.frame.to_ndarray(format="bgr24")


@dataclass(frozen=True)
class ReaderStatus:
    connected: bool
    fps: float
    last_frame_age_s: float | None
    lag_s: float | None  # how far decoding runs behind the stream's own timestamps
    width: int | None
    height: int | None
    codec: str | None
    reconnects: int
    frames: int
    last_error: str | None


class RtspReader(threading.Thread):
    """Reads one stream forever, reconnecting with a growing pause after each failure."""

    def __init__(self, camera_id: str, url: str, config: CaptureConfig) -> None:
        super().__init__(name=f"reader-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self._url = url
        self._config = config
        self._stop_event = threading.Event()
        self._lock = threading.Condition()
        self._latest: FramePacket | None = None
        self._seq = 0
        self._arrivals: deque[float] = deque(maxlen=120)
        self._info: tuple[int, int, str] | None = None
        self._reconnects = -1  # the first successful open is not a reconnect
        self._last_error: str | None = None
        self._lag_s: float | None = None
        # Smallest (arrival time - stream timestamp) seen in each recent second. The smallest
        # of those is what "on time" looks like; anything above it is backlog.
        self._offset_minima: deque[tuple[int, float]] = deque(maxlen=LAG_WINDOW_S)

    # -- consumer side ---------------------------------------------------------------

    def latest(self) -> FramePacket | None:
        with self._lock:
            return self._latest

    def wait_newer(self, seq: int, timeout: float) -> FramePacket | None:
        """Block until a frame newer than `seq` arrives, then return it. None on timeout."""
        with self._lock:
            ready = self._lock.wait_for(lambda: self._latest is not None and self._latest.seq > seq, timeout)
            return self._latest if ready else None

    def status(self) -> ReaderStatus:
        now = clock()
        with self._lock:
            latest = self._latest
            arrivals = [t for t in self._arrivals if now - t <= 2.0]
            info = self._info
            reconnects = max(self._reconnects, 0)
            error = self._last_error
            frames = self._seq
            lag = self._lag_s
        age = None if latest is None else now - latest.captured_at
        connected = age is not None and age <= self._config.stale_after_s
        return ReaderStatus(
            connected=connected,
            fps=round(len(arrivals) / 2.0, 1) if connected else 0.0,
            last_frame_age_s=None if age is None else round(age, 2),
            lag_s=None if lag is None or not connected else round(lag, 3),
            width=info[0] if info else None,
            height=info[1] if info else None,
            codec=info[2] if info else None,
            reconnects=reconnects,
            frames=frames,
            last_error=error,
        )

    def stop(self) -> None:
        self._stop_event.set()

    # -- reader thread ---------------------------------------------------------------

    def run(self) -> None:
        failures = 0
        backoff = self._config.reconnect_backoff_s or (1.0,)
        while not self._stop_event.is_set():
            frames_before = self._seq
            login_rejected = False
            try:
                self._read_stream()
                reason = "stream ended"
            except (av.error.HTTPUnauthorizedError, av.error.HTTPForbiddenError) as exc:
                login_rejected = True
                reason = mask_url(f"{type(exc).__name__}: {exc}")
            except Exception as exc:  # noqa: BLE001 - any failure means reconnect
                reason = mask_url(f"{type(exc).__name__}: {exc}")
            if self._stop_event.is_set():
                break
            # A session that delivered frames starts the backoff over.
            failures = 1 if self._seq > frames_before else failures + 1
            with self._lock:
                self._last_error = reason
            if login_rejected:
                # Hammering a camera with a wrong password gets the account locked.
                pause = self._config.auth_failure_pause_s
                log.error("camera_login_rejected", camera=self.camera_id, reason=reason, retry_in_s=pause)
            else:
                pause = backoff[min(failures - 1, len(backoff) - 1)]
                log.warning("camera_disconnected", camera=self.camera_id, reason=reason, retry_in_s=pause)
            self._stop_event.wait(pause)

    def _open(self) -> av.container.InputContainer:
        options = {"rtsp_transport": self._config.rtsp_transport} if self._url.lower().startswith("rtsp") else {}
        kwargs: dict = {"options": options, "timeout": (self._config.open_timeout_s, self._config.read_timeout_s)}
        if self._config.hwaccel != "none":
            from av.codec.hwaccel import HWAccel

            kwargs["hwaccel"] = HWAccel(device_type=self._config.hwaccel, allow_software_fallback=True)
        return av.open(self._url, **kwargs)

    def _measure_lag(self, now: float, stream_time: float) -> float:
        offset = now - stream_time
        second = int(now)
        if self._offset_minima and self._offset_minima[-1][0] == second:
            self._offset_minima[-1] = (second, min(self._offset_minima[-1][1], offset))
        else:
            self._offset_minima.append((second, offset))
        return offset - min(value for _, value in self._offset_minima)

    def _read_stream(self) -> None:
        with self._open() as container:
            stream = container.streams.video[0]
            context = stream.codec_context
            time_base = float(stream.time_base) if stream.time_base else None
            with self._lock:
                self._info = (context.width, context.height, context.name)
                self._reconnects += 1
                self._offset_minima.clear()
                self._lag_s = None
            log.info(
                "camera_connected",
                camera=self.camera_id,
                url=mask_url(self._url),
                codec=context.name,
                width=context.width,
                height=context.height,
            )
            bad_packets = 0
            for packet in container.demux(stream):
                if self._stop_event.is_set():
                    return
                try:
                    frames = packet.decode()
                except av.error.InvalidDataError:
                    # One damaged packet is not worth a reconnect; the next keyframe repairs it.
                    bad_packets += 1
                    if bad_packets > MAX_BAD_PACKETS:
                        raise
                    continue
                for frame in frames:
                    bad_packets = 0
                    now = clock()
                    lag = None
                    if frame.pts is not None and time_base is not None:
                        lag = self._measure_lag(now, frame.pts * time_base)
                    with self._lock:
                        self._seq += 1
                        self._latest = FramePacket(frame, self._seq, now, time.time())
                        self._arrivals.append(now)
                        self._last_error = None
                        self._lag_s = lag
                        self._lock.notify_all()
                    if lag is not None and lag > self._config.max_lag_s:
                        raise _FellBehindError(f"reader is {lag:.1f} s behind the stream")
