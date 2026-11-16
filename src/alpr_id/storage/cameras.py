"""Cameras added through the API, kept in the same SQLite file as the events.

Cameras from `config/app.yaml` are not stored here; the file stays their only source. The
URL column holds the camera password in clear text, like `.env` does, so the database file
needs the same care. The API never returns it unmasked.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from alpr_id.config import CameraConfig

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
    id          TEXT PRIMARY KEY,
    legacy_id   INTEGER,
    description TEXT NOT NULL,
    url         TEXT NOT NULL,
    roi         TEXT NOT NULL,
    process_fps REAL NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
"""


@dataclass(frozen=True)
class StoredCamera:
    camera: CameraConfig
    url: str


class CameraStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def all(self) -> list[StoredCamera]:
        """In the order the cameras were added."""
        with self._lock:
            # rowid breaks ties: the Windows clock ticks every 15.6 ms, so two cameras stored
            # right after each other can carry the same creation time.
            rows = self._conn.execute("SELECT * FROM cameras ORDER BY created_at, rowid").fetchall()
        return [self._to_camera(row) for row in rows]

    def get(self, camera_id: str) -> StoredCamera | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,)).fetchone()
        return self._to_camera(row) if row else None

    def upsert(self, camera: CameraConfig, url: str) -> None:
        """Insert, or replace everything except the creation time."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO cameras (id, legacy_id, description, url, roi, process_fps, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    legacy_id = excluded.legacy_id, description = excluded.description, url = excluded.url,
                    roi = excluded.roi, process_fps = excluded.process_fps, updated_at = excluded.updated_at
                """,
                (camera.id, camera.legacy_id, camera.description, url, json.dumps(list(camera.roi)), camera.process_fps,
                 now, now),
            )
            self._conn.commit()

    def delete(self, camera_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _to_camera(row: sqlite3.Row) -> StoredCamera:
        camera = CameraConfig(
            id=row["id"],
            legacy_id=row["legacy_id"],
            description=row["description"],
            roi=tuple(json.loads(row["roi"])),
            process_fps=row["process_fps"],
        )
        return StoredCamera(camera, row["url"])
