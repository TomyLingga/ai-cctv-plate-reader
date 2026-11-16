from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from alpr_id.pipeline.events import PlateEvent, iso_timestamp
from alpr_id.storage.db import EventStore
from alpr_id.storage.media import MediaStore

TZ = "Asia/Jakarta"


def make_event(event_id: str, camera: str, plate: str, epoch: float) -> PlateEvent:
    return PlateEvent(
        event_id=event_id,
        camera_id=camera,
        plate=plate,
        plate_formatted=plate,
        confidence=0.97,
        format_valid=True,
        timestamp=iso_timestamp(epoch, TZ),
        track_id=7,
        bbox=(10, 20, 110, 50),
        crop_url=f"/media/x/{event_id}_crop.jpg",
        region_known=True,
        alternatives=("BK1234HA",),
        reads=3,
        raw=plate,
        ts_epoch=epoch,
    )


def test_iso_timestamp_carries_the_utc_offset():
    assert iso_timestamp(1794560940.0, TZ).endswith("+07:00")


def test_event_roundtrip_and_filters(tmp_path):
    store = EventStore(tmp_path / "events.sqlite3")
    store.insert(make_event("a", "cam-1", "BK1234MA", 1000.0))
    store.insert(make_event("b", "cam-2", "BK8985HF", 2000.0))
    store.insert(make_event("c", "cam-1", "BM9719XX", 3000.0))

    assert store.count() == 3
    newest_first = store.query()
    assert [e["event_id"] for e in newest_first] == ["c", "b", "a"]

    first = newest_first[-1]
    assert first["plate"] == "BK1234MA"
    assert first["bbox"] == [10, 20, 110, 50]
    assert first["alternatives"] == ["BK1234HA"]
    assert first["format_valid"] is True and first["region_known"] is True
    assert first["timestamp"].endswith("+07:00")

    assert [e["event_id"] for e in store.query(camera="cam-1")] == ["c", "a"]
    assert [e["event_id"] for e in store.query(since_epoch=2000.0)] == ["c", "b"]
    assert [e["event_id"] for e in store.query(limit=1)] == ["c"]
    assert store.query(camera="cam-9") == []
    store.close()


def test_events_survive_reopening_the_database(tmp_path):
    path = tmp_path / "events.sqlite3"
    store = EventStore(path)
    store.insert(make_event("a", "cam-1", "BK1234MA", 1000.0))
    store.close()
    reopened = EventStore(path)
    assert reopened.count() == 1
    reopened.close()


def test_media_is_saved_under_date_and_camera(tmp_path):
    media = MediaStore(tmp_path / "media", TZ)
    image = np.random.default_rng(0).integers(0, 255, (40, 120, 3), dtype=np.uint8)
    epoch = datetime(2026, 11, 13, 9, 49, 0, tzinfo=ZoneInfo(TZ)).timestamp()
    url = media.save(image, "cam-2", epoch, "abcd1234_crop")
    assert url == "/media/2026-11-13/cam-2/094900_abcd1234_crop.jpg"
    saved = tmp_path / "media" / "2026-11-13" / "cam-2" / "094900_abcd1234_crop.jpg"
    assert saved.is_file() and saved.stat().st_size > 0


def test_purge_removes_only_old_day_folders(tmp_path):
    media = MediaStore(tmp_path / "media", TZ)
    today = datetime.now(ZoneInfo(TZ)).date()
    old = media.root / f"{today - timedelta(days=100):%Y-%m-%d}"
    recent = media.root / f"{today - timedelta(days=5):%Y-%m-%d}"
    other = media.root / "not-a-date"
    for folder in (old, recent, other):
        folder.mkdir()
    assert media.purge_older_than(90) == 1
    assert not old.exists() and recent.exists() and other.exists()
    assert media.purge_older_than(0) == 0
