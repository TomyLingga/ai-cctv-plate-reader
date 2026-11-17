"""Cameras added at run time: storage, URL checks, and the engine starting and stopping them."""
import threading
from types import SimpleNamespace
from typing import ClassVar

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from alpr_id.capture.probe import url_host, validate_stream_url
from alpr_id.capture.reader import ReaderStatus
from alpr_id.config import AppConfig, CameraConfig, StorageConfig
from alpr_id.logging_setup import mask_url
from alpr_id.pipeline import engine as engine_module
from alpr_id.pipeline.camera_pipeline import VisiblePlate
from alpr_id.pipeline.engine import CameraView, ConflictError, Engine, NotFoundError
from alpr_id.pipeline.events import PlateEvent
from alpr_id.storage.cameras import CameraStore
from alpr_id.storage.db import EventStore

URL = "rtsp://admin:rahasia@192.168.1.210:554/Streaming/Channels/101"


# -- storage -------------------------------------------------------------------------------


def test_stored_cameras_survive_reopening_and_keep_their_order(tmp_path):
    path = tmp_path / "db.sqlite3"
    store = CameraStore(path)
    store.upsert(CameraConfig(id="cam-b", legacy_id=6, description="kedua", roi=(0.1, 0.2, 0.9, 0.8), process_fps=5), URL)
    store.upsert(CameraConfig(id="cam-a"), URL + "?x=1")
    # Changing a camera does not move it in the list.
    store.upsert(CameraConfig(id="cam-b", description="diubah"), URL)
    store.close()

    reopened = CameraStore(path)
    stored = reopened.all()
    assert [item.camera.id for item in stored] == ["cam-b", "cam-a"]
    assert stored[0].camera == CameraConfig(id="cam-b", description="diubah")
    assert stored[0].url == URL
    assert reopened.get("cam-a").url == URL + "?x=1"
    assert reopened.get("cam-z") is None
    assert reopened.delete("cam-a") is True
    assert reopened.delete("cam-a") is False
    assert [item.camera.id for item in reopened.all()] == ["cam-b"]
    reopened.close()


# -- URL checks ----------------------------------------------------------------------------


@pytest.mark.parametrize("url", [URL, "rtsps://kamera.local/live", "  rtsp://192.168.1.5/ch1  ", "RTSP://10.0.0.2:8554/a"])
def test_rtsp_urls_are_accepted(url):
    assert validate_stream_url(url) == url.strip()


@pytest.mark.parametrize(
    "url",
    [
        "",
        "192.168.1.5",
        "http://192.168.1.5/video",
        "file:///C:/Windows/win.ini",
        "concat:a|b",
        "rtsp://",
        "rtsp://host:port/x",
        "rtsp://host/has space",
        "rtsp://host/" + "a" * 600,
    ],
)
def test_other_urls_are_rejected(url):
    with pytest.raises(ValueError):
        validate_stream_url(url)


@pytest.mark.parametrize(
    ("url", "shown"),
    [
        (URL, "rtsp://admin:***@192.168.1.210:554/Streaming/Channels/101"),
        ("rtsp://admin:p%40ss%3Aword@10.0.0.5/live", "rtsp://admin:***@10.0.0.5/live"),
        # An '@' typed into the password without encoding must not leak the rest of it.
        ("rtsp://admin:p@ss@10.0.0.5/live", "rtsp://admin:***@10.0.0.5/live"),
        ("rtsp://10.0.0.5/user=admin&password=rahasia&channel=1", "rtsp://10.0.0.5/user=admin&password=***&channel=1"),
        ("rtsp://10.0.0.5:554/live", "rtsp://10.0.0.5:554/live"),
        ("rtsp://viewer@10.0.0.5/live", "rtsp://viewer@10.0.0.5/live"),
    ],
)
def test_mask_url_hides_every_password(url, shown):
    assert mask_url(url) == shown
    # Error messages of the stream library quote the URL; it is masked there as well.
    assert mask_url(f"HTTPUnauthorizedError: [Errno 401] Server returned 401: '{url}'") == f"HTTPUnauthorizedError: [Errno 401] Server returned 401: '{shown}'"


def test_url_host():
    assert url_host(URL) == "192.168.1.210"
    assert url_host("rtsp://Kamera.Local:554/x") == "kamera.local"


# -- config --------------------------------------------------------------------------------


def test_config_file_cameras_need_url_env_and_a_safe_id():
    with pytest.raises(ValidationError, match="url_env"):
        AppConfig(cameras=(CameraConfig(id="cam-1"),))
    for bad in ("", "cam 1", "../x", "a" * 33, "-cam"):
        with pytest.raises(ValidationError):
            CameraConfig(id=bad, url_env="X")


# -- engine --------------------------------------------------------------------------------


class FakeReader(threading.Thread):
    """Reader that never connects. Records which URLs were opened and which were stopped."""

    opened: ClassVar[list[tuple[str, str]]] = []
    stopped: ClassVar[list[str]] = []

    def __init__(self, camera_id, url, config):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self._halt = threading.Event()
        FakeReader.opened.append((camera_id, url))

    def run(self):
        self._halt.wait()

    def stop(self):
        FakeReader.stopped.append(self.camera_id)
        self._halt.set()

    def latest(self):
        return None

    def wait_newer(self, seq, timeout):
        self._halt.wait(0.01)  # no frame ever arrives

    def status(self):
        return ReaderStatus(connected=False, fps=0.0, last_frame_age_s=None, lag_s=None, width=None, height=None,
                            codec=None, reconnects=0, frames=0, last_error=None)


@pytest.fixture
def make_engine(tmp_path, monkeypatch):
    FakeReader.opened, FakeReader.stopped = [], []
    models = (SimpleNamespace(providers=["CPUExecutionProvider"]), SimpleNamespace(alphabet="AB_", pad_char="_"))
    monkeypatch.setattr(engine_module, "load_models", lambda config: models)
    monkeypatch.setattr(engine_module, "RtspReader", FakeReader)
    monkeypatch.setenv("TEST_CAM_1_URL", "rtsp://admin:satu@192.168.1.205/x")
    config = AppConfig(
        cameras=(CameraConfig(id="cam-1", legacy_id=1, description="dari berkas", url_env="TEST_CAM_1_URL"),),
        storage=StorageConfig(database=tmp_path / "db.sqlite3", media_dir=tmp_path / "media"),
    )
    engines = []

    def start() -> Engine:
        engine = Engine(config)
        engine.start()
        engines.append(engine)
        return engine

    yield start
    for engine in engines:
        if engine.store is not None:
            engine.stop()


def ids(engine: Engine) -> list[str]:
    return [camera["id"] for camera in engine.cameras()]


def test_added_camera_runs_at_once_and_after_a_restart(make_engine):
    engine = make_engine()
    assert ids(engine) == ["cam-1"]
    added = engine.add_camera(CameraConfig(id="cam-5", legacy_id=5, description="baru", process_fps=6), URL)
    assert (added["id"], added["origin"], added["process_fps"], added["connected"]) == ("cam-5", "ui", 6.0, False)
    assert added["current"] == {"camera_id": "cam-5", "visible": False, "plate": None}
    assert ids(engine) == ["cam-1", "cam-5"]
    assert engine.has_camera("cam-5")
    assert ("cam-5", URL) in FakeReader.opened
    # The legacy routes find it by its number, although it is not in the config file.
    assert engine.camera_id_for_legacy(5) == "cam-5"
    assert engine.legacy_plate(5) == {"plate": "", "raw": "", "fps": 0.0, "status": "offline"}
    assert engine.health()["cameras_total"] == 2
    engine.stop()
    engine.store = None  # already stopped; the fixture must not stop it again

    again = make_engine()
    assert ids(again) == ["cam-1", "cam-5"]
    assert {c["id"]: c["origin"] for c in again.cameras()} == {"cam-1": "config", "cam-5": "ui"}


def test_camera_sources_never_show_a_password(make_engine):
    engine = make_engine()
    engine.add_camera(CameraConfig(id="cam-5"), URL)
    assert engine.camera_sources() == [
        {"id": "cam-1", "origin": "config", "url": "rtsp://admin:***@192.168.1.205/x", "url_env": "TEST_CAM_1_URL"},
        {"id": "cam-5", "origin": "ui", "url": "rtsp://admin:***@192.168.1.210:554/Streaming/Channels/101", "url_env": None},
    ]
    assert engine.stream_url("cam-5") == URL
    assert "rahasia" not in str(engine.cameras())


def test_duplicates_are_refused(make_engine):
    engine = make_engine()
    with pytest.raises(ConflictError) as error:
        engine.add_camera(CameraConfig(id="cam-1"), URL)
    assert error.value.code == "duplicate_id"
    with pytest.raises(ConflictError) as error:
        engine.add_camera(CameraConfig(id="cam-5", legacy_id=1), URL)
    assert error.value.code == "duplicate_legacy_id"
    assert ids(engine) == ["cam-1"]
    assert engine.camera_store.all() == []


def test_camera_limit(make_engine, monkeypatch):
    monkeypatch.setattr(engine_module, "MAX_CAMERAS", 2)
    engine = make_engine()
    engine.add_camera(CameraConfig(id="cam-5"), URL)
    with pytest.raises(ConflictError) as error:
        engine.add_camera(CameraConfig(id="cam-6"), URL)
    assert error.value.code == "camera_limit"


def test_config_cameras_cannot_be_changed_or_removed(make_engine):
    engine = make_engine()
    with pytest.raises(ConflictError) as error:
        engine.update_camera(CameraConfig(id="cam-1", description="x"), None)
    assert error.value.code == "config_camera"
    with pytest.raises(ConflictError):
        engine.remove_camera("cam-1")
    with pytest.raises(NotFoundError):
        engine.remove_camera("cam-9")
    assert ids(engine) == ["cam-1"]


def test_update_restarts_the_stream_only_when_it_has_to(make_engine):
    engine = make_engine()
    engine.add_camera(CameraConfig(id="cam-5", description="lama"), URL)
    FakeReader.opened.clear()

    # A new description or legacy number leaves the stream alone.
    updated = engine.update_camera(CameraConfig(id="cam-5", description="baru", legacy_id=7), None)
    assert (updated["description"], updated["legacy_id"]) == ("baru", 7)
    assert FakeReader.opened == []
    assert FakeReader.stopped == []
    assert engine.camera_id_for_legacy(7) == "cam-5"

    # A new area, rate or URL needs a new reader; the old one is stopped.
    updated = engine.update_camera(CameraConfig(id="cam-5", description="baru", roi=(0.1, 0.1, 0.9, 0.9)), None)
    assert updated["roi"] == [0.1, 0.1, 0.9, 0.9]
    assert FakeReader.opened == [("cam-5", URL)]  # the stored URL was kept
    assert FakeReader.stopped == ["cam-5"]

    new_url = URL.replace("101", "102")
    engine.update_camera(CameraConfig(id="cam-5", roi=(0.1, 0.1, 0.9, 0.9)), new_url)
    assert FakeReader.opened[-1] == ("cam-5", new_url)
    assert engine.camera_store.get("cam-5").url == new_url
    assert ids(engine) == ["cam-1", "cam-5"]

    with pytest.raises(ConflictError) as error:
        engine.update_camera(CameraConfig(id="cam-5", legacy_id=1), None)
    assert error.value.code == "duplicate_legacy_id"


def test_removed_camera_stops_and_stays_gone(make_engine):
    engine = make_engine()
    engine.add_camera(CameraConfig(id="cam-5", legacy_id=5), URL)
    engine.remove_camera("cam-5")
    assert ids(engine) == ["cam-1"]
    assert FakeReader.stopped == ["cam-5"]
    assert not engine.has_camera("cam-5")
    assert engine.legacy_plate(5) is None
    with pytest.raises(KeyError):
        engine.current("cam-5")
    with pytest.raises(KeyError):
        engine.video_jpeg("cam-5")
    engine.stop()
    engine.store = None
    assert ids(make_engine()) == ["cam-1"]


def test_stored_camera_that_collides_with_the_config_file_is_not_started(make_engine, tmp_path):
    store = CameraStore(tmp_path / "db.sqlite3")
    store.upsert(CameraConfig(id="cam-1", description="tabrakan id"), URL)
    store.upsert(CameraConfig(id="cam-6", legacy_id=1, description="tabrakan nomor"), URL)
    store.upsert(CameraConfig(id="cam-7"), URL)
    store.close()
    engine = make_engine()
    assert ids(engine) == ["cam-1", "cam-7"]
    assert engine.cameras()[0]["description"] == "dari berkas"


def test_no_signal_picture_follows_the_requested_width(make_engine):
    engine = make_engine()
    for width in (640, 1280):
        image = cv2.imdecode(np.frombuffer(engine.video_jpeg("cam-1", width), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert image.shape[:2] == (width * 9 // 16, width)


def test_picture_of_a_part_of_the_frame_keeps_the_plate_box_on_the_plate():
    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    plate = VisiblePlate(
        track_id=1, box=(1200, 700, 1320, 740), plate=None, plate_formatted=None, raw=None, confidence=None,
        format_valid=None, region_known=None, alternatives=(), reads=0, event_id=None, last_seen=engine_module.clock(),
    )
    view = CameraView(seq=1, frame=frame, visible=(plate,))

    # The middle quarter of the frame is 1280 x 720 pixels, and is shown without shrinking.
    part = Engine._render(view, 1280, (0.25, 0.25, 0.75, 0.75))
    assert part.shape[:2] == (720, 1280)
    green = np.argwhere(part[:, :, 1] == 255)
    assert tuple(green.min(axis=0)) == (700 - 360 - 1, 1200 - 640 - 1)  # box line, 2 px thick, in crop pixels
    assert not frame.any()  # drawn on a copy, the frame the pipeline keeps is untouched

    # Asked for narrower than the part is: scaled down, the box with it.
    small = Engine._render(view, 640, (0.25, 0.25, 0.75, 0.75))
    assert small.shape[:2] == (360, 640)
    assert small[:, :, 1].max() > 200


# -- region codes in the engine ------------------------------------------------------------


def test_engine_starts_with_the_built_in_region_codes(make_engine):
    engine = make_engine()
    listed = engine.region_list()
    assert len(listed) == 61
    assert {"code": "BK", "name": "Sumatera Utara (Medan dan pesisir timur)", "builtin": True, "seen": 0} in listed
    assert "BK" in engine.regions
    assert "BF" not in engine.regions


def test_region_changes_reach_the_running_cameras_and_survive_a_restart(make_engine):
    engine = make_engine()
    pipeline_regions = engine._workers["cam-1"].pipeline._regions
    assert pipeline_regions is engine.regions  # one list for every camera

    added = engine.add_region("XK", "Kode uji")
    assert added == {"code": "XK", "name": "Kode uji", "builtin": False, "seen": 0}
    assert "XK" in pipeline_regions
    engine.remove_region("DK")
    assert "DK" not in pipeline_regions

    with pytest.raises(ConflictError) as error:
        engine.add_region("XK", "lagi")
    assert error.value.code == "duplicate_code"
    with pytest.raises(ConflictError) as error:
        engine.add_region("CD", "diplomatik")
    assert error.value.code == "special_code"
    with pytest.raises(NotFoundError) as error:
        engine.remove_region("DK")
    assert error.value.code == "unknown_code"

    engine.stop()
    engine.store = None
    again = make_engine()
    assert "XK" in again.regions
    assert "DK" not in again.regions
    assert again.restore_regions() == 1
    assert "DK" in again.regions
    assert len(again.region_list()) == 62


def test_engine_learns_from_stored_events_which_codes_pass_here(make_engine, tmp_path):
    store = EventStore(tmp_path / "db.sqlite3")
    for n, plate in enumerate(["BK8857JL", "BK8962LL", "BK8250GP", "BM9719XX"]):
        store.insert(PlateEvent(
            event_id=f"e{n}", camera_id="cam-1", plate=plate, plate_formatted=plate, confidence=0.9, format_valid=True,
            timestamp="2026-11-13T21:00:00+07:00", track_id=n, bbox=(1, 2, 3, 4), region_known=True, ts_epoch=float(n),
        ))
    # A code that was replaced is not evidence of what passes here.
    store.insert(PlateEvent(
        event_id="fixed", camera_id="cam-1", plate="BK8418FD", plate_formatted="BK 8418 FD", confidence=0.6,
        format_valid=True, timestamp="2026-11-13T21:14:59+07:00", track_id=9, bbox=(1, 2, 3, 4), region_known=True,
        raw="BF8418FD", ts_epoch=9.0, region_corrected=True,
    ))
    store.close()

    engine = make_engine()
    assert engine.regions.seen() == {"BK": 3, "BM": 1}
    assert {row["code"]: row["seen"] for row in engine.region_list() if row["seen"]} == {"BK": 3, "BM": 1}
    assert engine.cameras()[0]["plates_in_view"] == 0
