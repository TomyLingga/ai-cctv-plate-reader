"""Plate crops and frame snapshots on disk, served under /media."""
from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from alpr_id.logging_setup import get_logger

log = get_logger(__name__)

MEDIA_URL_PREFIX = "/media"


class MediaStore:
    """Layout: <media_dir>/<YYYY-MM-DD>/<camera>/<HHMMSS>_<id>_<kind>.jpg"""

    def __init__(self, media_dir: str | Path, timezone: str) -> None:
        self._root = Path(media_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._tz = ZoneInfo(timezone)

    @property
    def root(self) -> Path:
        return self._root

    def save(self, image: np.ndarray, camera_id: str, epoch_seconds: float, name: str, quality: int = 92) -> str | None:
        """Write a JPEG and return its URL, or None when writing failed."""
        moment = datetime.fromtimestamp(epoch_seconds, self._tz)
        relative = Path(f"{moment:%Y-%m-%d}") / camera_id / f"{moment:%H%M%S}_{name}.jpg"
        target = self._root / relative
        try:
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise ValueError("JPEG encoding failed")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(encoded.tobytes())
        except (OSError, ValueError, cv2.error) as exc:
            log.error("media_write_failed", path=str(target), error=str(exc))
            return None
        return f"{MEDIA_URL_PREFIX}/{relative.as_posix()}"

    def purge_older_than(self, days: int) -> int:
        """Delete day folders older than `days`. Returns how many were removed."""
        if days <= 0:
            return 0
        cutoff = (datetime.now(self._tz) - timedelta(days=days)).date()
        removed = 0
        for folder in self._root.iterdir():
            if not folder.is_dir():
                continue
            try:
                day = datetime.strptime(folder.name, "%Y-%m-%d").date()
            except ValueError:
                continue
            if day < cutoff:
                shutil.rmtree(folder, ignore_errors=True)
                removed += 1
        return removed
