"""Plate text recognition with a fast-plate-ocr model, returning full probabilities.

`LicensePlateRecognizer.run` only reports the winning character and its probability per slot.
Voting needs the probability of every character, so this wrapper uses the library to load
the model and its plate config and to resize crops, then calls the ONNX session directly.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fast_plate_ocr import LicensePlateRecognizer
from fast_plate_ocr.core.process import preprocess_image, resize_image

from alpr_id.postprocess.voting import OcrRead


class PlateOcr:
    def __init__(self, model_path: str | Path, plate_config_path: str | Path, providers: list) -> None:
        for path in (model_path, plate_config_path):
            if not Path(path).is_file():
                raise FileNotFoundError(f"OCR file not found: {path}. Run scripts/fetch_models.py or see README.")
        self._recognizer = LicensePlateRecognizer(
            onnx_model_path=str(model_path), plate_config_path=str(plate_config_path), providers=providers
        )
        self._config = self._recognizer.config
        self._session = self._recognizer.model
        self._output = self._recognizer.plate_output_name
        self._input = self._session.get_inputs()[0].name

    @property
    def alphabet(self) -> str:
        return self._config.alphabet

    @property
    def pad_char(self) -> str:
        return self._config.pad_char

    @property
    def providers(self) -> list[str]:
        return self._session.get_providers()

    def _prepare(self, crop_bgr: np.ndarray) -> np.ndarray:
        if self._config.image_color_mode == "rgb":
            image = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        resized = resize_image(
            image,
            self._config.img_height,
            self._config.img_width,
            image_color_mode=self._config.image_color_mode,
            keep_aspect_ratio=self._config.keep_aspect_ratio,
            interpolation_method=self._config.interpolation,
            padding_color=self._config.padding_color,
        )
        return resized if resized.ndim == 3 else resized[..., None]

    def read(self, crops_bgr: list[np.ndarray]) -> list[OcrRead]:
        """One OCR pass over all crops as a single batch."""
        if not crops_bgr:
            return []
        batch = preprocess_image(np.stack([self._prepare(crop) for crop in crops_bgr]))
        output = self._session.run([self._output], {self._input: batch})[0]
        probs = output.reshape((-1, self._config.max_plate_slots, len(self._config.alphabet)))
        return [OcrRead.from_probs(p, self._config.alphabet, self._config.pad_char) for p in probs]
