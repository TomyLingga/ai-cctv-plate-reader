"""The region codes in use, and the choice of the nearest one for a code that is not listed.

One `RegionBook` is shared by every camera. The list can be replaced while the cameras run,
and the book counts which codes pass by, so that the code most often seen at this site
wins when two listed codes fit a read about equally well.
"""
from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Iterable, Mapping

import numpy as np

from alpr_id.postprocess.plate_format import REGION_CODES


class RegionBook:
    def __init__(self, codes: Iterable[str] = REGION_CODES, local_boost: float = 2.0) -> None:
        """`local_boost` is how much more weight the most common code of this site can get.

        A code that made up every vehicle so far counts `1 + local_boost` times, a code never
        seen counts once. The limit keeps a clear OCR result from being overruled by habit.
        """
        self._codes = frozenset(codes)
        self._boost = local_boost
        self._seen: Counter[str] = Counter()
        self._lock = threading.Lock()

    def __contains__(self, code: object) -> bool:
        return code in self._codes

    def __len__(self) -> int:
        return len(self._codes)

    @property
    def codes(self) -> frozenset[str]:
        return self._codes

    def replace(self, codes: Iterable[str]) -> None:
        """Swap in a new list. Readers see either the old or the new one, never a mix."""
        self._codes = frozenset(codes)

    def observe(self, code: str, count: int = 1) -> None:
        """Record vehicles whose region code was read as it is, without replacement."""
        with self._lock:
            self._seen[code] += count

    def seen(self) -> dict[str, int]:
        with self._lock:
            return dict(self._seen)

    def _share(self, code: str) -> float:
        with self._lock:
            total = sum(self._seen.values())
            return self._seen[code] / total if total else 0.0

    def nearest(self, slot_probs: np.ndarray, index: Mapping[str, int]) -> tuple[str, float] | None:
        """The listed code that the OCR output supports best, with its probability.

        `slot_probs` holds one row per character of the code as it was read; only listed codes
        of that length are compared, because a code of another length would shift every later
        character. Returns None when no listed code has that length or none has any support.
        """
        length = slot_probs.shape[0]
        best: tuple[float, float, str] | None = None
        for code in sorted(self._codes):
            if len(code) != length:
                continue
            probability = 1.0
            for position, char in enumerate(code):
                column = index.get(char)
                probability *= float(slot_probs[position, column]) if column is not None else 0.0
            score = probability * (1.0 + self._boost * self._share(code))
            if score > 0 and (best is None or score > best[0]):
                best = (score, probability, code)
        return (best[2], best[1]) if best else None
