"""Event history in SQLite."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from alpr_id.pipeline.events import PlateEvent
from alpr_id.postprocess.plate_format import parse_standard

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    camera_id       TEXT NOT NULL,
    plate           TEXT NOT NULL,
    plate_formatted TEXT NOT NULL,
    confidence      REAL NOT NULL,
    format_valid    INTEGER NOT NULL,
    region_known    INTEGER NOT NULL,
    alternatives    TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    ts_epoch        REAL NOT NULL,
    track_id        INTEGER NOT NULL,
    bbox            TEXT NOT NULL,
    crop_url        TEXT,
    snapshot_url    TEXT,
    reads           INTEGER NOT NULL,
    raw             TEXT NOT NULL,
    region_corrected INTEGER NOT NULL DEFAULT 0,
    stopped         INTEGER
);
CREATE INDEX IF NOT EXISTS idx_events_camera_time ON events (camera_id, ts_epoch);
CREATE INDEX IF NOT EXISTS idx_events_time ON events (ts_epoch);
"""

# Columns added after the first release, for databases created before them.
_ADDED_COLUMNS = {"region_corrected": "INTEGER NOT NULL DEFAULT 0", "stopped": "INTEGER"}

_COLUMNS = (
    "event_id", "camera_id", "plate", "plate_formatted", "confidence", "format_valid", "region_known", "alternatives",
    "timestamp", "ts_epoch", "track_id", "bbox", "crop_url", "snapshot_url", "reads", "raw", "region_corrected", "stopped",
)


class EventStore:
    """One connection shared by all threads, every call serialised by a lock.

    Write volume is a few events per minute, so one connection is enough. WAL keeps the
    database consistent if the process is killed mid-write.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        present = {row["name"] for row in self._conn.execute("PRAGMA table_info(events)")}
        for name, definition in _ADDED_COLUMNS.items():
            if name not in present:
                self._conn.execute(f"ALTER TABLE events ADD COLUMN {name} {definition}")
        self._conn.commit()

    def insert(self, event: PlateEvent) -> None:
        with self._lock:
            self._conn.execute(
                f"INSERT INTO events ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' * len(_COLUMNS))})",
                (
                    event.event_id,
                    event.camera_id,
                    event.plate,
                    event.plate_formatted,
                    event.confidence,
                    int(event.format_valid),
                    int(event.region_known),
                    json.dumps(list(event.alternatives)),
                    event.timestamp,
                    event.ts_epoch,
                    event.track_id,
                    json.dumps(list(event.bbox)),
                    event.crop_url,
                    event.snapshot_url,
                    event.reads,
                    event.raw,
                    int(event.region_corrected),
                    None if event.stopped is None else int(event.stopped),
                ),
            )
            self._conn.commit()

    def region_counts(self, limit: int = 5000) -> dict[str, int]:
        """How often each region code was read, as it is, among the newest `limit` events."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT plate FROM events WHERE format_valid = 1 AND region_known = 1 AND region_corrected = 0 "
                "ORDER BY ts_epoch DESC LIMIT ?",
                (limit,),
            ).fetchall()
        counts: dict[str, int] = {}
        for row in rows:
            parts = parse_standard(row["plate"])
            if parts is not None:
                counts[parts.prefix] = counts.get(parts.prefix, 0) + 1
        return counts

    def query(self, camera: str | None = None, since_epoch: float | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Newest first."""
        clauses, params = [], []
        if camera:
            clauses.append("camera_id = ?")
            params.append(camera)
        if since_epoch is not None:
            clauses.append("ts_epoch >= ?")
            params.append(since_epoch)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM events {where} ORDER BY ts_epoch DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(sql, (*params, limit)).fetchall()
        return [self._to_dict(row) for row in rows]

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "event_id": row["event_id"],
            "camera_id": row["camera_id"],
            "plate": row["plate"],
            "plate_formatted": row["plate_formatted"],
            "confidence": round(row["confidence"], 4),
            "format_valid": bool(row["format_valid"]),
            "timestamp": row["timestamp"],
            "track_id": row["track_id"],
            "bbox": json.loads(row["bbox"]),
            "crop_url": row["crop_url"],
            "snapshot_url": row["snapshot_url"],
            "region_known": bool(row["region_known"]),
            "alternatives": json.loads(row["alternatives"]),
            "reads": row["reads"],
            "raw": row["raw"],
            "region_corrected": bool(row["region_corrected"]),
            "stopped": None if row["stopped"] is None else bool(row["stopped"]),
        }
