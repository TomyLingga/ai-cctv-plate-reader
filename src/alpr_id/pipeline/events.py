"""Plate events and the bus that carries them from camera threads to API subscribers."""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class PlateEvent:
    """One vehicle seen by one camera."""

    event_id: str
    camera_id: str
    plate: str
    plate_formatted: str
    confidence: float
    format_valid: bool
    timestamp: str  # ISO 8601 with UTC offset
    track_id: int
    bbox: tuple[int, int, int, int]
    crop_url: str | None = None
    snapshot_url: str | None = None
    region_known: bool = False
    alternatives: tuple[str, ...] = ()
    reads: int = 0
    raw: str = ""
    ts_epoch: float = 0.0
    # The region code read was not on the list and was replaced by the nearest listed one.
    region_corrected: bool = False
    # Whether the vehicle stood still when it was read. False for a vehicle read in passing.
    # None for events stored before this was recorded.
    stopped: bool | None = None

    def with_media(self, crop_url: str | None, snapshot_url: str | None) -> PlateEvent:
        return replace(self, crop_url=crop_url, snapshot_url=snapshot_url)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "plate": self.plate,
            "plate_formatted": self.plate_formatted,
            "confidence": round(self.confidence, 4),
            "format_valid": self.format_valid,
            "timestamp": self.timestamp,
            "track_id": self.track_id,
            "bbox": list(self.bbox),
            "crop_url": self.crop_url,
            "snapshot_url": self.snapshot_url,
            "region_known": self.region_known,
            "alternatives": list(self.alternatives),
            "reads": self.reads,
            "raw": self.raw,
            "region_corrected": self.region_corrected,
            "stopped": self.stopped,
        }


def iso_timestamp(epoch_seconds: float, timezone: str) -> str:
    return datetime.fromtimestamp(epoch_seconds, ZoneInfo(timezone)).isoformat(timespec="seconds")


class EventBus:
    """Fan-out of events to asyncio subscribers. `publish` may be called from any thread."""

    def __init__(self, queue_size: int = 100) -> None:
        self._queue_size = queue_size
        self._subscribers: dict[asyncio.Queue[dict[str, Any]], asyncio.AbstractEventLoop] = {}
        self._lock = threading.Lock()

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Call from inside the event loop that will read the queue."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._queue_size)
        with self._lock:
            self._subscribers[queue] = asyncio.get_running_loop()
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._subscribers.pop(queue, None)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def publish(self, payload: dict[str, Any]) -> None:
        with self._lock:
            targets = list(self._subscribers.items())
        for queue, loop in targets:
            try:
                loop.call_soon_threadsafe(self._offer, queue, payload)
            except RuntimeError:  # loop already closed
                self.unsubscribe(queue)

    @staticmethod
    def _offer(queue: asyncio.Queue[dict[str, Any]], payload: dict[str, Any]) -> None:
        # A subscriber that stopped reading loses its oldest event instead of blocking the rest.
        if queue.full():
            queue.get_nowait()
        queue.put_nowait(payload)
