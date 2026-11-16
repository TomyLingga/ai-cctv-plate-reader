"""Region of interest: the part of the frame where plates can appear."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PixelRect:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def crop(self, image: np.ndarray) -> np.ndarray:
        """View into `image`, no copy."""
        return image[self.y1 : self.y2, self.x1 : self.x2]


def roi_to_pixels(roi: tuple[float, float, float, float], width: int, height: int) -> PixelRect:
    """Turn a fractional [x1, y1, x2, y2] into pixel coordinates clamped to the frame."""
    x1, y1, x2, y2 = roi
    px1 = min(max(int(round(x1 * width)), 0), width - 1)
    py1 = min(max(int(round(y1 * height)), 0), height - 1)
    px2 = min(max(int(round(x2 * width)), px1 + 1), width)
    py2 = min(max(int(round(y2 * height)), py1 + 1), height)
    return PixelRect(px1, py1, px2, py2)


def clamp_box(box: tuple[float, float, float, float], width: int, height: int, padding: float = 0.0) -> PixelRect | None:
    """Pixel rectangle for a detection box, grown by `padding` of its size on every side."""
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    px1 = max(int(round(x1 - pad_x)), 0)
    py1 = max(int(round(y1 - pad_y)), 0)
    px2 = min(int(round(x2 + pad_x)), width)
    py2 = min(int(round(y2 + pad_y)), height)
    if px2 - px1 < 2 or py2 - py1 < 2:
        return None
    return PixelRect(px1, py1, px2, py2)
