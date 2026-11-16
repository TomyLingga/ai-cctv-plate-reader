"""Per-stage latency measurement."""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np


def clock() -> float:
    """Monotonic seconds with sub-millisecond resolution.

    `time.monotonic()` ticks every 15.6 ms on Windows with Python 3.12, too coarse for frame
    timestamps and for scheduling eight frames per second.
    """
    return time.perf_counter()


class StageTimer:
    """Collects how long each named stage took, in milliseconds. Safe to share across threads."""

    def __init__(self, keep: int = 5000) -> None:
        self._keep = keep
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - started) * 1000.0)

    def add(self, name: str, milliseconds: float) -> None:
        with self._lock:
            samples = self._samples[name]
            samples.append(milliseconds)
            if len(samples) > self._keep:
                del samples[: len(samples) - self._keep]

    def reset(self) -> None:
        """Forget everything measured so far, e.g. after a warm-up period."""
        with self._lock:
            self._samples.clear()

    def summary(self) -> dict[str, dict[str, float]]:
        """Count, mean, median, 95th percentile and maximum per stage."""
        with self._lock:
            snapshot = {name: list(values) for name, values in self._samples.items()}
        out: dict[str, dict[str, float]] = {}
        for name, values in snapshot.items():
            if not values:
                continue
            arr = np.asarray(values)
            out[name] = {
                "count": int(arr.size),
                "mean_ms": round(float(arr.mean()), 2),
                "p50_ms": round(float(np.percentile(arr, 50)), 2),
                "p95_ms": round(float(np.percentile(arr, 95)), 2),
                "max_ms": round(float(arr.max()), 2),
            }
        return out
