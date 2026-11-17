"""API contract, with a stand-in engine so no camera or model is needed."""
import asyncio
import threading

import cv2
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from alpr_id.api.app import create_app, parse_since
from alpr_id.capture.probe import ProbeResult
from alpr_id.config import AppConfig, CameraConfig, StorageConfig
from alpr_id.pipeline.engine import ConflictError, NotFoundError
from alpr_id.pipeline.events import EventBus

TOKEN = "token-untuk-test"
ADMIN = {"Authorization": f"Bearer {TOKEN}"}
SECRET_URL = "rtsp://admin:rahasia123@192.168.1.210:554/Streaming/Channels/101"

EVENT = {
    "event_id": "abc123",
    "camera_id": "cam-1",
    "plate": "BK1234ABC",
    "plate_formatted": "BK 1234 ABC",
    "confidence": 0.97,
    "format_valid": True,
    "timestamp": "2026-11-13T09:49:00+07:00",
    "track_id": 42,
    "bbox": [10, 20, 110, 50],
    "crop_url": "/media/2026-11-13/cam-1/094900_abc123_crop.jpg",
}


class FakeStore:
    def __init__(self):
        self.calls = []

    def query(self, camera=None, since_epoch=None, limit=100):
        self.calls.append((camera, since_epoch, limit))
        return [EVENT]


class FakeEngine:
    def __init__(self):
        self.store = FakeStore()
        self.bus = EventBus()
        self.recognized_shape = None
        self.added = {}  # cameras added through the API: id -> (config, url)
        self.jpeg_calls = []
        self.regions = {"BK": "Sumatera Utara"}  # region codes: code -> area

    def health(self):
        return {"status": "ok", "cameras_total": 1, "cameras_connected": 1}

    def cameras(self):
        return [{"id": "cam-1", "legacy_id": 1, "connected": True, "fps": 20.0}]

    def has_camera(self, camera_id):
        return camera_id == "cam-1"

    def current(self, camera_id):
        if camera_id != "cam-1":
            raise KeyError(camera_id)
        return {"camera_id": camera_id, "visible": True, "plate": "BK1234ABC"}

    def recognize(self, image):
        self.recognized_shape = image.shape
        return [{"plate": "BK1234ABC", "format_valid": True}]

    def legacy_plate(self, legacy_id):
        return {"plate": "BK 1234 ABC", "raw": "BK1234ABC", "fps": 20.0, "status": "online"} if legacy_id == 1 else None

    def camera_id_for_legacy(self, legacy_id):
        return "cam-1" if legacy_id == 1 else None

    # -- what the camera routes need ---------------------------------------------------

    def video_jpeg(self, camera_id, width=None, crop=None):
        if camera_id != "cam-1":
            raise KeyError(camera_id)
        self.jpeg_calls.append(width if crop is None else (width, crop))
        return b"\xff\xd8jpeg"

    def add_camera(self, camera, url):
        if camera.id == "cam-1":
            raise ConflictError("duplicate_id", "camera id cam-1 is already in use")
        self.added[camera.id] = (camera, url)
        return {"id": camera.id, "origin": "ui", "legacy_id": camera.legacy_id}

    def update_camera(self, camera, url):
        if camera.id == "cam-1":
            raise ConflictError("config_camera", "camera cam-1 is defined in the config file")
        if camera.id not in self.added:
            raise NotFoundError("unknown_camera", f"unknown camera {camera.id}")
        self.added[camera.id] = (camera, url or self.added[camera.id][1])
        return {"id": camera.id, "origin": "ui", "description": camera.description}

    def remove_camera(self, camera_id):
        if camera_id not in self.added:
            raise NotFoundError("unknown_camera", f"unknown camera {camera_id}")
        del self.added[camera_id]

    def camera_sources(self):
        return [{"id": "cam-1", "origin": "config", "url": "rtsp://admin:***@192.168.1.205:554/x", "url_env": "X"}]

    def stream_url(self, camera_id):
        if camera_id != "cam-1":
            raise KeyError(camera_id)
        return "rtsp://admin:tersimpan@192.168.1.205:554/x"

    # -- what the region routes need ---------------------------------------------------

    def region_list(self):
        return [{"code": code, "name": name, "builtin": code == "BK", "seen": 12 if code == "BK" else 0}
                for code, name in sorted(self.regions.items())]

    def add_region(self, code, name):
        if code == "CD":
            raise ConflictError("special_code", "CD is a diplomatic code")
        if code in self.regions:
            raise ConflictError("duplicate_code", f"region code {code} is already listed")
        self.regions[code] = name
        return {"code": code, "name": name, "builtin": False, "seen": 0}

    def remove_region(self, code):
        if code not in self.regions:
            raise NotFoundError("unknown_code", f"region code {code} is not listed")
        del self.regions[code]

    def restore_regions(self):
        self.regions.setdefault("DK", "Bali")
        return 1


class FakeProbe:
    """Stands in for the real connection test; answers with whatever `result` is set to."""

    def __init__(self):
        self.urls = []
        self.result = ProbeResult("ok", codec="hevc", width=2560, height=1440, fps=20.0,
                                  frame=np.zeros((1440, 2560, 3), dtype=np.uint8))

    def __call__(self, url, capture):
        self.urls.append(url)
        return self.result


def make_client(tmp_path, legacy_routes=True, admin_token=TOKEN, probe=None):
    config = AppConfig(
        cameras=(CameraConfig(id="cam-1", legacy_id=1, url_env="X"),),
        storage=StorageConfig(database=tmp_path / "db.sqlite3", media_dir=tmp_path / "media"),
        api={"legacy_routes": legacy_routes},
    )
    engine = FakeEngine()
    app = create_app(config, engine=engine, manage_engine=False, admin_token=admin_token, probe=probe or FakeProbe())
    return TestClient(app), engine


def test_health_and_cameras(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/health").json()["status"] == "ok"
    cameras = client.get("/cameras").json()
    assert cameras[0]["id"] == "cam-1" and cameras[0]["connected"] is True


def test_current_plate_and_unknown_camera(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/cameras/cam-1/current").json()["plate"] == "BK1234ABC"
    assert client.get("/cameras/cam-9/current").status_code == 404


def test_events_passes_filters_to_the_store(tmp_path):
    client, engine = make_client(tmp_path)
    response = client.get("/events", params={"camera": "cam-1", "since": "2026-11-13T09:00:00+07:00", "limit": 5})
    assert response.status_code == 200
    assert response.json() == [EVENT]
    camera, since_epoch, limit = engine.store.calls[-1]
    assert (camera, limit) == ("cam-1", 5)
    assert since_epoch == pytest.approx(1794535200.0)

    client.get("/events")
    assert engine.store.calls[-1] == (None, None, 100)


def test_events_rejects_bad_parameters(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.get("/events", params={"since": "kemarin"}).status_code == 400
    assert client.get("/events", params={"limit": 0}).status_code == 422
    assert client.get("/events", params={"limit": 5000}).status_code == 422


def test_parse_since_accepts_epoch_iso_and_naive_local_time():
    assert parse_since(None, "Asia/Jakarta") is None
    assert parse_since("1794535200", "Asia/Jakarta") == 1794535200.0
    assert parse_since("2026-11-13T09:00:00+07:00", "Asia/Jakarta") == pytest.approx(1794535200.0)
    assert parse_since("2026-11-13T09:00:00", "Asia/Jakarta") == pytest.approx(1794535200.0)
    # An unencoded '+' in a query string arrives as a space.
    assert parse_since("2026-11-13T09:00:00 07:00", "Asia/Jakarta") == pytest.approx(1794535200.0)
    with pytest.raises(HTTPException):
        parse_since("not a date", "Asia/Jakarta")


def test_recognize_decodes_the_uploaded_image(tmp_path):
    client, engine = make_client(tmp_path)
    ok, png = cv2.imencode(".png", np.zeros((120, 320, 3), dtype=np.uint8))
    assert ok
    response = client.post("/recognize", files={"file": ("plate.png", png.tobytes(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert (body["width"], body["height"]) == (320, 120)
    assert body["results"][0]["plate"] == "BK1234ABC"
    assert engine.recognized_shape == (120, 320, 3)


def test_recognize_rejects_a_file_that_is_not_an_image(tmp_path):
    client, _ = make_client(tmp_path)
    response = client.post("/recognize", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_legacy_plate_route_keeps_the_old_shape(tmp_path):
    client, _ = make_client(tmp_path)
    response = client.get("/plate/1")
    assert response.status_code == 200
    assert response.json() == {"plate": "BK 1234 ABC", "raw": "BK1234ABC", "fps": 20.0, "status": "online"}
    assert client.get("/plate/9").status_code == 404
    assert client.get("/video/9").status_code == 404


def test_legacy_routes_can_be_switched_off(tmp_path):
    client, _ = make_client(tmp_path, legacy_routes=False)
    assert client.get("/plate/1").status_code == 404
    assert client.get("/health").status_code == 200


def test_media_files_are_served(tmp_path):
    client, _ = make_client(tmp_path)
    folder = tmp_path / "media" / "2026-11-13" / "cam-1"
    folder.mkdir(parents=True)
    (folder / "crop.jpg").write_bytes(b"jpegbytes")
    response = client.get("/media/2026-11-13/cam-1/crop.jpg")
    assert response.status_code == 200 and response.content == b"jpegbytes"


def test_event_bus_delivers_from_another_thread():
    async def scenario():
        bus = EventBus(queue_size=2)
        queue = bus.subscribe()
        assert bus.subscriber_count == 1
        thread = threading.Thread(target=lambda: [bus.publish({"n": n}) for n in range(3)])
        thread.start()
        thread.join()
        await asyncio.sleep(0.05)
        # The queue holds two; the oldest of three was dropped instead of blocking the publisher.
        received = [queue.get_nowait()["n"] for _ in range(queue.qsize())]
        bus.unsubscribe(queue)
        return received, bus.subscriber_count

    received, remaining = asyncio.run(scenario())
    assert received == [1, 2]
    assert remaining == 0
