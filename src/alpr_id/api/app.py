"""REST API: status, current plates, event history, live event stream, still-image recognition,
camera management, and the two browser pages built on top of them.

Also serves the two routes of the service this project replaces (`/plate/{n}`, `/video/{n}`)
so the weighbridge desktop app and the registration web app keep working.
"""
from __future__ import annotations

import base64
import hmac
import os
import re
import threading
import time
from collections import deque
from collections.abc import AsyncIterable, Callable, Iterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from alpr_id import __version__
from alpr_id.capture.probe import ProbeResult, probe_stream, url_host, validate_stream_url
from alpr_id.config import CAMERA_ID_PATTERN, AppConfig, CameraConfig, CaptureConfig, Roi, check_roi
from alpr_id.logging_setup import get_logger, mask_url
from alpr_id.pipeline.engine import Engine, ManageError, NotFoundError
from alpr_id.postprocess.plate_format import REGION_CODE_PATTERN
from alpr_id.storage.media import MEDIA_URL_PREFIX
from alpr_id.timing import clock

log = get_logger(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
STATIC_DIR = Path(__file__).parent / "static"
ADMIN_TOKEN_ENV = "ALPR_ADMIN_TOKEN"
MIN_ADMIN_TOKEN_LENGTH = 8
SNAPSHOT_WIDTH_STEP = 160
TEST_SNAPSHOT_WIDTH = 960

Probe = Callable[[str, CaptureConfig], ProbeResult]


class _CameraFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(default="", max_length=120)
    legacy_id: int | None = Field(default=None, ge=1, le=999)
    roi: Roi = (0.0, 0.0, 1.0, 1.0)
    process_fps: float = Field(default=8.0, ge=1, le=15)

    @field_validator("roi")
    @classmethod
    def _roi_inside_frame(cls, roi: Roi) -> Roi:
        return check_roi(roi)

    @field_validator("description")
    @classmethod
    def _trim(cls, value: str) -> str:
        return value.strip()


class CameraCreate(_CameraFields):
    id: str = Field(pattern=CAMERA_ID_PATTERN)
    url: str

    @field_validator("url")
    @classmethod
    def _rtsp_only(cls, url: str) -> str:
        return validate_stream_url(url)


class CameraUpdate(_CameraFields):
    # Left out or empty: keep the URL that is stored.
    url: str | None = None

    @field_validator("url")
    @classmethod
    def _rtsp_only(cls, url: str | None) -> str | None:
        return validate_stream_url(url) if url and url.strip() else None


class CameraTest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str | None = None
    # Test the stored URL of this camera instead, for a camera whose URL is not retyped.
    camera_id: str | None = None

    @field_validator("url")
    @classmethod
    def _rtsp_only(cls, url: str | None) -> str | None:
        return validate_stream_url(url) if url and url.strip() else None


class RegionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    name: str = Field(default="", max_length=120)

    @field_validator("code")
    @classmethod
    def _one_or_two_letters(cls, code: str) -> str:
        code = code.strip().upper()
        if not REGION_CODE_PATTERN.fullmatch(code):
            raise ValueError("a region code is one or two letters")
        return code

    @field_validator("name")
    @classmethod
    def _trim(cls, value: str) -> str:
        return value.strip()


class LoginGuard:
    """Refuses connection tests to a camera that has just rejected the login a few times.

    Cameras lock the account after several wrong passwords in a row, and a locked account
    would also cut off the stream that is already running.
    """

    def __init__(self, max_failures: int = 3, window_s: float = 600.0) -> None:
        self._max = max_failures
        self._window = window_s
        self._failures: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def retry_after(self, host: str) -> float:
        """Seconds until a test of `host` is allowed again; 0 when it is allowed now."""
        now = clock()
        with self._lock:
            recent = self._failures.get(host)
            if not recent:
                return 0.0
            while recent and now - recent[0] > self._window:
                recent.popleft()
            if len(recent) < self._max:
                return 0.0
            return max(self._window - (now - recent[0]), 1.0)

    def record_rejection(self, host: str) -> None:
        with self._lock:
            self._failures.setdefault(host, deque(maxlen=self._max)).append(clock())

    def clear(self, host: str) -> None:
        with self._lock:
            self._failures.pop(host, None)


def parse_since(value: str | None, timezone: str) -> float | None:
    """Accept seconds since the epoch or an ISO 8601 date-time; naive times use `timezone`."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        pass
    # An unencoded '+' in a query string arrives as a space: '...T09:00:00 07:00'.
    value = re.sub(r" (\d{2}:?\d{2})$", r"+\1", value)
    try:
        moment = datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="since must be ISO 8601 or seconds since the epoch") from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=ZoneInfo(timezone))
    return moment.timestamp()


def _error(status: int, code: str, message: str, headers: dict[str, str] | None = None) -> HTTPException:
    """Errors of the camera routes carry a stable `code` next to the English message."""
    return HTTPException(status_code=status, detail={"code": code, "message": message}, headers=headers)


def _manage_error(exc: ManageError) -> HTTPException:
    return _error(404 if isinstance(exc, NotFoundError) else 409, exc.code, exc.message)


class _PageFiles(StaticFiles):
    """Scripts and styles of the browser pages.

    `no-cache` makes the browser ask whether its copy is still current before using it, so
    an updated service never runs with the scripts of the previous version.
    """

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app(
    config: AppConfig,
    engine: Engine | None = None,
    manage_engine: bool = True,
    admin_token: str | None = None,
    probe: Probe = probe_stream,
) -> FastAPI:
    """Build the app. With `manage_engine` the engine starts and stops with the server.

    `admin_token` protects the routes that change cameras. Left as None it is read from the
    environment variable ALPR_ADMIN_TOKEN; without a token those routes refuse every request.
    """
    engine = engine or Engine(config)
    token = (admin_token if admin_token is not None else os.environ.get(ADMIN_TOKEN_ENV, "")).strip()
    if token and len(token) < MIN_ADMIN_TOKEN_LENGTH:
        log.warning("admin_token_too_short", minimum=MIN_ADMIN_TOKEN_LENGTH)
        token = ""
    guard = LoginGuard()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if manage_engine:
            engine.start()
        try:
            yield
        finally:
            if manage_engine:
                engine.stop()

    app = FastAPI(title="alpr-id", version=__version__, lifespan=lifespan)
    app.state.engine = engine

    config.storage.media_dir.mkdir(parents=True, exist_ok=True)
    app.mount(MEDIA_URL_PREFIX, StaticFiles(directory=str(config.storage.media_dir)), name="media")

    def _authorized(authorization: str | None) -> bool:
        scheme, _, supplied = (authorization or "").partition(" ")
        return bool(token) and scheme.lower() == "bearer" and hmac.compare_digest(supplied.strip().encode(), token.encode())

    def require_admin(authorization: str | None = Header(default=None)) -> None:
        if not token:
            raise _error(403, "admin_disabled", f"set {ADMIN_TOKEN_ENV} in .env and restart to allow camera changes")
        if not _authorized(authorization):
            raise _error(401, "bad_token", "missing or wrong admin token", {"WWW-Authenticate": "Bearer"})

    admin = [Depends(require_admin)]

    # -- status and readings -----------------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, Any]:
        return engine.health()

    @app.get("/cameras")
    def cameras() -> list[dict[str, Any]]:
        return engine.cameras()

    @app.get("/cameras/{camera_id}/current")
    async def current(camera_id: str) -> dict[str, Any]:
        try:
            return engine.current(camera_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown camera {camera_id}") from None

    @app.get("/cameras/{camera_id}/snapshot.jpg")
    def snapshot(camera_id: str, width: int = Query(default=640, ge=160, le=1920)) -> Response:
        """Latest processed frame with the plate boxes drawn. One JPEG per request."""
        # Round to a few fixed widths: every distinct width is rendered and cached separately.
        width = max(round(width / SNAPSHOT_WIDTH_STEP), 1) * SNAPSHOT_WIDTH_STEP
        try:
            jpeg = engine.video_jpeg(camera_id, width)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown camera {camera_id}") from None
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/cameras/{camera_id}/video")
    def video(camera_id: str) -> StreamingResponse:
        """MJPEG of one camera, for an <img>. Same picture as /video/{n} of the legacy routes."""
        if not engine.has_camera(camera_id):
            raise HTTPException(status_code=404, detail=f"unknown camera {camera_id}")
        return _mjpeg_response(engine, camera_id, config.api.video_fps)

    @app.get("/events")
    def events(
        camera: str | None = None,
        since: str | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[dict[str, Any]]:
        return engine.store.query(camera or None, parse_since(since, config.timezone), limit)

    @app.get("/events/stream", response_class=EventSourceResponse)
    async def stream() -> AsyncIterable[ServerSentEvent]:
        queue = engine.bus.subscribe()
        try:
            while True:
                payload = await queue.get()
                yield ServerSentEvent(data=payload, event="plate", id=payload["event_id"])
        finally:
            engine.bus.unsubscribe(queue)

    @app.post("/recognize")
    def recognize(file: UploadFile) -> dict[str, Any]:
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="image larger than 20 MB")
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="file is not a readable image")
        return {"width": int(image.shape[1]), "height": int(image.shape[0]), "results": engine.recognize(image)}

    # -- camera management, admin token required ---------------------------------------

    @app.get("/admin/check")
    def admin_check(authorization: str | None = Header(default=None)) -> dict[str, bool]:
        """Lets the browser page tell 'no token set on the server' from 'wrong token'."""
        return {"configured": bool(token), "authorized": _authorized(authorization)}

    @app.get("/admin/cameras", dependencies=admin)
    def admin_cameras() -> list[dict[str, Any]]:
        """Where each camera comes from and its URL with the password hidden."""
        return engine.camera_sources()

    @app.post("/cameras", status_code=201, dependencies=admin)
    def add_camera(body: CameraCreate) -> dict[str, Any]:
        camera = CameraConfig(
            id=body.id, legacy_id=body.legacy_id, description=body.description, roi=body.roi, process_fps=body.process_fps
        )
        try:
            return engine.add_camera(camera, body.url)
        except ManageError as exc:
            raise _manage_error(exc) from None

    @app.put("/cameras/{camera_id}", dependencies=admin)
    def update_camera(camera_id: str, body: CameraUpdate) -> dict[str, Any]:
        if not re.fullmatch(CAMERA_ID_PATTERN, camera_id):
            raise _error(404, "unknown_camera", f"unknown camera {camera_id}")
        camera = CameraConfig(
            id=camera_id, legacy_id=body.legacy_id, description=body.description, roi=body.roi,
            process_fps=body.process_fps,
        )
        try:
            return engine.update_camera(camera, body.url)
        except ManageError as exc:
            raise _manage_error(exc) from None

    @app.delete("/cameras/{camera_id}", status_code=204, dependencies=admin)
    def remove_camera(camera_id: str) -> Response:
        try:
            engine.remove_camera(camera_id)
        except ManageError as exc:
            raise _manage_error(exc) from None
        return Response(status_code=204)

    @app.post("/cameras/test", dependencies=admin)
    def test_camera(body: CameraTest) -> dict[str, Any]:
        """Open the URL once and return what the camera delivers, with one frame as a data URL."""
        url = body.url
        if url is None and body.camera_id:
            try:
                url = engine.stream_url(body.camera_id)
            except KeyError:
                raise _error(404, "unknown_camera", f"unknown camera {body.camera_id}") from None
        if not url:
            raise _error(422, "url_required", "give a url, or the camera_id of a stored camera")
        host = url_host(url)
        wait = guard.retry_after(host)
        if wait > 0:
            raise _error(
                429, "login_locked",
                f"{host} rejected the login several times; wait before trying again so the camera does not lock the account",
                {"Retry-After": str(int(wait))},
            )
        result = probe(url, config.capture)
        if result.code == "login_rejected":
            guard.record_rejection(host)
        elif result.ok:
            guard.clear(host)
        log.info("camera_tested", url=mask_url(url), result=result.code)
        payload: dict[str, Any] = {
            "ok": result.ok,
            "code": result.code,
            "codec": result.codec,
            "width": result.width,
            "height": result.height,
            "fps": result.fps,
            "detail": result.detail,
            "snapshot": None,
        }
        if result.frame is not None:
            payload["snapshot"] = _data_url(result.frame, TEST_SNAPSHOT_WIDTH)
        return payload

    # -- region codes: reading is open, changing needs the admin token -------------------

    @app.get("/regions")
    def regions() -> list[dict[str, Any]]:
        """The region codes accepted as known, with how often each was read at this site."""
        return engine.region_list()

    @app.post("/regions", status_code=201, dependencies=admin)
    def add_region(body: RegionCreate) -> dict[str, Any]:
        try:
            return engine.add_region(body.code, body.name)
        except ManageError as exc:
            raise _manage_error(exc) from None

    @app.post("/regions/restore", dependencies=admin)
    def restore_regions() -> dict[str, int]:
        """Put back the built-in codes that were removed."""
        return {"added": engine.restore_regions()}

    @app.delete("/regions/{code}", status_code=204, dependencies=admin)
    def remove_region(code: str) -> Response:
        try:
            engine.remove_region(code.strip().upper())
        except ManageError as exc:
            raise _manage_error(exc) from None
        return Response(status_code=204)

    # -- browser pages -----------------------------------------------------------------

    def page(name: str) -> FileResponse:
        return FileResponse(STATIC_DIR / name, media_type="text/html", headers={"Cache-Control": "no-cache"})

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui")

    @app.get("/ui", include_in_schema=False)
    def ui_dashboard() -> FileResponse:
        return page("index.html")

    @app.get("/ui/cameras", include_in_schema=False)
    def ui_cameras() -> FileResponse:
        return page("cameras.html")

    @app.get("/ui/regions", include_in_schema=False)
    def ui_regions() -> FileResponse:
        return page("regions.html")

    app.mount("/ui/static", _PageFiles(directory=str(STATIC_DIR)), name="ui-static")

    # -- routes of the old service -----------------------------------------------------

    if config.api.legacy_routes:

        # async on purpose: it only reads memory, and must not wait for a threadpool slot
        # held by video viewers while the desktop app polls it every second.
        @app.get("/plate/{legacy_id}")
        async def legacy_plate(legacy_id: int) -> dict[str, Any]:
            payload = engine.legacy_plate(legacy_id)
            if payload is None:
                raise HTTPException(status_code=404, detail=f"unknown camera {legacy_id}")
            return payload

        @app.get("/video/{legacy_id}")
        def legacy_video(legacy_id: int) -> StreamingResponse:
            camera_id = engine.camera_id_for_legacy(legacy_id)
            if camera_id is None:
                raise HTTPException(status_code=404, detail=f"unknown camera {legacy_id}")
            return _mjpeg_response(engine, camera_id, config.api.video_fps)

    return app


def _data_url(frame: np.ndarray, max_width: int) -> str | None:
    height, width = frame.shape[:2]
    scale = min(max_width / width, 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii") if ok else None


def _mjpeg_response(engine: Engine, camera_id: str, fps: float) -> StreamingResponse:
    return StreamingResponse(
        _mjpeg(engine, camera_id, 1.0 / fps),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


def _mjpeg(engine: Engine, camera_id: str, interval: float) -> Iterator[bytes]:
    """Endless JPEG stream. The generator is closed by the server when the client leaves."""
    while True:
        started = time.monotonic()
        try:
            jpeg = engine.video_jpeg(camera_id)
        except KeyError:
            return  # the camera was removed while someone was watching
        if jpeg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(max(interval - (time.monotonic() - started), 0.01))
