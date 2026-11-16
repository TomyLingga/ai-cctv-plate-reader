"""The list of region codes the service accepts, kept in the same SQLite file as the events.

A new database starts with the built-in list (postprocess/plate_format.py). After that the
table is the only source: codes can be added and removed while the service runs, and a
removed built-in code stays removed until the defaults are restored.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from alpr_id.postprocess.plate_format import REGION_NAMES

_SCHEMA = """
CREATE TABLE IF NOT EXISTS region_codes (
    code       TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    builtin    INTEGER NOT NULL,
    created_at REAL NOT NULL
);
"""


class RegionStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        existed = self._conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'region_codes'").fetchone()
        self._conn.executescript(_SCHEMA)
        if not existed:
            self._insert_defaults()
        self._conn.commit()

    def _insert_defaults(self) -> int:
        """Add every built-in code that is missing. Existing rows are left as they are."""
        now = time.time()
        before = self._conn.total_changes
        self._conn.executemany(
            "INSERT OR IGNORE INTO region_codes (code, name, builtin, created_at) VALUES (?, ?, 1, ?)",
            [(code, name, now) for code, name in REGION_NAMES.items()],
        )
        return self._conn.total_changes - before

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT code, name, builtin FROM region_codes ORDER BY code").fetchall()
        return [{"code": row["code"], "name": row["name"], "builtin": bool(row["builtin"])} for row in rows]

    def codes(self) -> frozenset[str]:
        with self._lock:
            return frozenset(row["code"] for row in self._conn.execute("SELECT code FROM region_codes"))

    def add(self, code: str, name: str) -> bool:
        """False when the code is already listed."""
        with self._lock:
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO region_codes (code, name, builtin, created_at) VALUES (?, ?, ?, ?)",
                (code, name, int(code in REGION_NAMES), time.time()),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def remove(self, code: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM region_codes WHERE code = ?", (code,))
            self._conn.commit()
            return cursor.rowcount > 0

    def restore_defaults(self) -> int:
        """Put back the built-in codes that were removed. Returns how many came back."""
        with self._lock:
            added = self._insert_defaults()
            self._conn.commit()
            return added

    def close(self) -> None:
        with self._lock:
            self._conn.close()
