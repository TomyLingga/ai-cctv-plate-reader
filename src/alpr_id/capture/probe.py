"""Check a camera URL before it is saved: open it once and decode one frame."""
from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import av
import numpy as np

from alpr_id.config import CaptureConfig
from alpr_id.logging_setup import mask_url

ALLOWED_SCHEMES = ("rtsp", "rtsps")
MAX_URL_LENGTH = 500


def validate_stream_url(url: str) -> str:
    """Return the trimmed URL, or raise ValueError saying what is wrong with it.

    Only RTSP is accepted. PyAV would happily open files and other protocols too, and this
    URL comes from a browser.
    """
    url = url.strip()
    if not url:
        raise ValueError("url is empty")
    if len(url) > MAX_URL_LENGTH:
        raise ValueError(f"url is longer than {MAX_URL_LENGTH} characters")
    if any(ch.isspace() or ord(ch) < 32 for ch in url):
        raise ValueError("url must not contain spaces or control characters")
    try:
        parts = urlsplit(url)
        host = parts.hostname
        _ = parts.port  # raises ValueError for a port that is not a number
    except ValueError as exc:
        raise ValueError("url is not a valid address") from exc
    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError("url must start with rtsp:// or rtsps://")
    if not host:
        raise ValueError("url has no host")
    return url


def url_host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


@dataclass(frozen=True)
class ProbeResult:
    """`code` is one of: ok, login_rejected, not_found, unreachable, timeout, no_video, error."""

    code: str
    codec: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frame: np.ndarray | None = None
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.code == "ok"


def _classify(exc: Exception) -> str:
    if isinstance(exc, (av.error.HTTPUnauthorizedError, av.error.HTTPForbiddenError)):
        return "login_rejected"
    if isinstance(exc, av.error.HTTPNotFoundError):
        return "not_found"
    # PyAV raises ExitError when its own open or read timeout fires.
    if isinstance(exc, (TimeoutError, av.error.ExitError)):
        return "timeout"
    if isinstance(exc, (ConnectionError, OSError)):
        return "unreachable"
    return "error"


def probe_stream(url: str, config: CaptureConfig, deadline_s: float = 20.0) -> ProbeResult:
    """Open `url` once and decode the first frame. Never raises; the result says what happened."""
    options = {"rtsp_transport": config.rtsp_transport} if url.lower().startswith("rtsp") else {}
    started = time.monotonic()
    try:
        container = av.open(url, options=options, timeout=(config.open_timeout_s, config.read_timeout_s))
    except Exception as exc:  # noqa: BLE001 - every failure is reported to the caller
        return ProbeResult(_classify(exc), detail=mask_url(f"{type(exc).__name__}: {exc}"))
    try:
        if not container.streams.video:
            return ProbeResult("no_video", detail="the stream has no video track")
        stream = container.streams.video[0]
        context = stream.codec_context
        fps = float(stream.average_rate) if stream.average_rate else None
        info = {"codec": context.name, "width": context.width, "height": context.height, "fps": fps}
        for packet in container.demux(stream):
            try:
                frames = packet.decode()
            except av.error.InvalidDataError:
                frames = []  # the stream was joined between keyframes; the next one repairs it
            for frame in frames:
                return ProbeResult("ok", frame=frame.to_ndarray(format="bgr24"), **info)
            if time.monotonic() - started > deadline_s:
                break
        return ProbeResult("timeout", detail="connected, but no frame could be decoded in time", **info)
    except Exception as exc:  # noqa: BLE001
        return ProbeResult(_classify(exc), detail=mask_url(f"{type(exc).__name__}: {exc}"))
    finally:
        container.close()
