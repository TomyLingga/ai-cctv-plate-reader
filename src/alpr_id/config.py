"""Application configuration: YAML file for settings, environment for secrets.

Camera URLs carry credentials, so the YAML only names the environment variable that holds
each URL. `.env` is loaded into the environment at startup.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_CONFIG_PATH = "config/app.yaml"

# The id ends up in API paths and in media folder names, so it has to be safe in both.
CAMERA_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$"

Roi = tuple[float, float, float, float]


def check_roi(roi: Roi) -> Roi:
    x1, y1, x2, y2 = roi
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("roi must be [x1, y1, x2, y2] as fractions with x1 < x2 and y1 < y2")
    return roi


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CameraConfig(_Section):
    id: str = Field(pattern=CAMERA_ID_PATTERN)
    legacy_id: int | None = None
    description: str = ""
    # Name of the environment variable that holds the URL. Empty for cameras added through
    # the API, whose URL is kept in the database instead.
    url_env: str = ""
    roi: Roi = (0.0, 0.0, 1.0, 1.0)
    # A second, small search area inside the frame, read on top of `roi`: the spot where
    # vehicles stand. The detector shrinks `roi` to its input size, and a plate that ends up
    # too small there is missed; the same plate in a crop the size of the detector input
    # (640 x 640 pixels) is found. Neither view finds every plate, both together find more
    # (docs/fase-2-baseline-dan-benchmark.md, section 3d). None = `roi` only.
    focus: Roi | None = None
    process_fps: float = Field(default=8.0, gt=0)

    @field_validator("roi")
    @classmethod
    def _roi_inside_frame(cls, roi: Roi) -> Roi:
        return check_roi(roi)

    @field_validator("focus")
    @classmethod
    def _focus_inside_frame(cls, focus: Roi | None) -> Roi | None:
        return check_roi(focus) if focus is not None else None

    @model_validator(mode="after")
    def _focus_inside_roi(self) -> CameraConfig:
        # A plate found outside `roi` would count as cut by the border and never be read.
        if self.focus is not None:
            (x1, y1, x2, y2), (fx1, fy1, fx2, fy2) = self.roi, self.focus
            if not (x1 <= fx1 and y1 <= fy1 and fx2 <= x2 and fy2 <= y2):
                raise ValueError("focus must lie inside roi")
        return self


class CaptureConfig(_Section):
    rtsp_transport: Literal["tcp", "udp"] = "tcp"
    open_timeout_s: float = 10.0
    read_timeout_s: float = 5.0
    reconnect_backoff_s: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)
    stale_after_s: float = 3.0
    # A rejected login is not retried soon: cameras lock the account after repeated failures.
    auth_failure_pause_s: float = 600.0
    # Reconnect when the reader has fallen this far behind the stream.
    max_lag_s: float = 2.0
    hwaccel: Literal["none", "d3d11va", "cuda"] = "none"


class RuntimeConfig(_Section):
    providers: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")
    gpu_mem_limit_mb: int | None = 1536
    # One inference call at a time across all cameras. Keeps GPU memory and CPU threads
    # predictable; cameras skip frames instead of queueing when the models are busy.
    serialize_inference: bool = True


class DetectorConfig(_Section):
    model_path: Path = Path("models/detector/plat.onnx")
    conf_threshold: float = Field(default=0.35, ge=0, le=1)
    iou_threshold: float = Field(default=0.5, ge=0, le=1)


class TrackingConfig(_Section):
    # How long the tracker keeps a plate it no longer sees before giving it a new id.
    lost_track_s: float = Field(default=4.0, gt=0)
    track_activation_threshold: float = 0.5
    high_conf_det_threshold: float = 0.5
    minimum_consecutive_frames: int = 2
    # A track not seen for this long is closed and its event, if any, is emitted.
    track_timeout_s: float = 5.0


class OcrConfig(_Section):
    model_path: Path = Path("models/ocr/cct_s_v2_global.onnx")
    plate_config_path: Path = Path("models/ocr/cct_s_v2_global_plate_config.yaml")
    max_reads_per_track: int = Field(default=5, ge=1)
    read_interval_s: float = Field(default=0.4, ge=0)
    # After the first reads, a crop is read only if it is this much better than the best one
    # read so far, or if nothing better turned up for stale_read_s.
    read_improvement: float = Field(default=0.25, ge=0)
    stale_read_s: float = Field(default=3.0, gt=0)
    # A vehicle whose reads gave nothing usable is read afresh after this long, with a new
    # budget of max_reads_per_track. 0 switches the second attempt off.
    retry_after_s: float = Field(default=8.0, ge=0)
    # When a plate is read:
    #   stopped             only while it stands still, so the reads and the picture stored
    #                       with the event are sharp. A vehicle that never stops in view is
    #                       not reported by this camera.
    #   stopped_or_passing  as stopped, and a vehicle that leaves without ever standing still
    #                       gets one read from its best frame. Standing reads come first:
    #                       that passing read is thrown away when this camera saw a standing
    #                       vehicle within passing_quiet_s before the passing one appeared or
    #                       after it left, because it is then the same truck rolling in or
    #                       out. It also has to reach voting.min_passing_confidence.
    #   always              as soon as it is seen, moving or not.
    read_when: Literal["stopped", "stopped_or_passing", "always"] = "stopped_or_passing"
    passing_quiet_s: float = Field(default=20.0, ge=0)
    # Standing still means: within this time the plate moved less than still_tolerance of
    # its own width, and its width changed less than twice that.
    still_window_s: float = Field(default=0.75, gt=0)
    still_tolerance: float = Field(default=0.12, gt=0, le=1)
    min_plate_width_px: int = 60
    crop_padding: float = Field(default=0.03, ge=0, le=0.5)
    crop_bottom_ratio: float = Field(default=0.0, ge=0, lt=0.6)
    # Kept as an explicit switch so the decision is visible in the config: crops are never
    # sharpened, thresholded, denoised or upscaled before OCR.
    enhance: Literal[False] = False


class PostprocessConfig(_Section):
    confusable_letters: tuple[tuple[str, ...], ...] = (("H", "M", "W"),)
    ambiguity_margin: float = Field(default=0.15, ge=0, le=1)
    # A region code that is not on the list is replaced by the listed code the reads support
    # best. The event keeps the string as read in `raw` and is marked `region_corrected`.
    fix_unknown_region: bool = True
    # How much extra weight the code most often seen at this site gets in that choice: a code
    # carried by every vehicle so far counts 1 + this value, a code never seen counts 1.
    region_local_boost: float = Field(default=2.0, ge=0)


class VotingConfig(_Section):
    min_reads: int = Field(default=2, ge=1)
    emit_confidence: float = Field(default=0.85, ge=0, le=1)
    # Every character must reach this before an event is emitted early.
    min_char_confidence: float = Field(default=0.6, ge=0, le=1)
    min_event_confidence: float = Field(default=0.4, ge=0, le=1)
    # A vehicle read only in passing has one frame and no second opinion, so the bar is
    # higher, and the result has to fit the standard plate pattern.
    min_passing_confidence: float = Field(default=0.8, ge=0, le=1)
    dedup_window_s: float = 120.0


class StorageConfig(_Section):
    database: Path = Path("data/alpr.sqlite3")
    media_dir: Path = Path("data/media")
    save_snapshot: bool = True
    # Also keep every crop that went through OCR, with its raw read. Source material for
    # the training dataset.
    save_reads: bool = True
    retention_days: int = 90


class ApiConfig(_Section):
    legacy_routes: bool = True
    legacy_plate_format: Literal["spaced", "compact"] = "spaced"
    legacy_hold_s: float = 5.0
    video_width: int = 1280
    video_fps: float = 8.0


class AppConfig(_Section):
    timezone: str = "Asia/Jakarta"
    cameras: tuple[CameraConfig, ...]
    capture: CaptureConfig = CaptureConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    detector: DetectorConfig = DetectorConfig()
    tracking: TrackingConfig = TrackingConfig()
    ocr: OcrConfig = OcrConfig()
    postprocess: PostprocessConfig = PostprocessConfig()
    voting: VotingConfig = VotingConfig()
    storage: StorageConfig = StorageConfig()
    api: ApiConfig = ApiConfig()

    @model_validator(mode="after")
    def _unique_ids(self) -> AppConfig:
        ids = [camera.id for camera in self.cameras]
        if len(set(ids)) != len(ids):
            raise ValueError("camera ids must be unique")
        missing = [camera.id for camera in self.cameras if not camera.url_env]
        if missing:
            raise ValueError(f"cameras in the config file need url_env: {', '.join(missing)}")
        legacy = [camera.legacy_id for camera in self.cameras if camera.legacy_id is not None]
        if len(set(legacy)) != len(legacy):
            raise ValueError("camera legacy_id values must be unique")
        return self

    def camera(self, camera_id: str) -> CameraConfig | None:
        return next((camera for camera in self.cameras if camera.id == camera_id), None)

    def camera_by_legacy_id(self, legacy_id: int) -> CameraConfig | None:
        return next((camera for camera in self.cameras if camera.legacy_id == legacy_id), None)


def load_config(path: str | os.PathLike[str] | None = None, env_file: str | os.PathLike[str] | None = ".env") -> AppConfig:
    """Load `.env` into the environment, then read and validate the YAML config."""
    if env_file and Path(env_file).is_file():
        load_dotenv(env_file, override=False)
    config_path = Path(path or os.environ.get("ALPR_CONFIG", DEFAULT_CONFIG_PATH))
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}. Copy config/app.example.yaml to config/app.yaml.")
    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return AppConfig.model_validate(data)


def camera_url(camera: CameraConfig) -> str:
    """URL of a camera from the environment. Raises when the variable is missing."""
    url = os.environ.get(camera.url_env, "").strip()
    if not url:
        raise KeyError(f"Environment variable {camera.url_env} for camera {camera.id} is not set (see .env.example)")
    return url
