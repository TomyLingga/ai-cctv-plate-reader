"""Crop quality score: which frame of a vehicle is worth running OCR on."""
from __future__ import annotations

import cv2
import numpy as np

_REFERENCE_SIZE = (160, 80)  # width, height the crop is brought to before measuring sharpness
_SHARPNESS_HALF_POINT = 150.0  # Laplacian variance at which the sharpness term reaches 0.5
_FULL_WIDTH_PX = 160.0  # crops at least this wide get the full size term


def sharpness(crop_bgr: np.ndarray) -> float:
    """Variance of the Laplacian on a size-normalised grayscale crop. Higher is sharper."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, _REFERENCE_SIZE, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_quality(crop_bgr: np.ndarray) -> float:
    """Score in 0..1 combining sharpness and size. Used to rank crops of one track."""
    if crop_bgr.size == 0:
        return 0.0
    sharp = sharpness(crop_bgr)
    sharp_term = sharp / (sharp + _SHARPNESS_HALF_POINT)
    size_term = min(crop_bgr.shape[1] / _FULL_WIDTH_PX, 1.0)
    return sharp_term * size_term
