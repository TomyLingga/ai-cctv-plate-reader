"""Camera management routes, the admin token, and the browser pages."""
import time

import pytest
from test_api import ADMIN, SECRET_URL, TOKEN, FakeProbe, make_client

from alpr_id.api.app import LoginGuard
from alpr_id.capture.probe import ProbeResult

NEW_CAMERA = {
    "id": "cam-5",
    "description": " Timbangan 3 masuk ",
    "url": SECRET_URL,
    "legacy_id": 5,
    "roi": [0.1, 0.2, 0.9, 0.8],
    "process_fps": 6,
}


def test_camera_changes_are_refused_without_a_token_on_the_server(tmp_path):
    client, engine = make_client(tmp_path, admin_token="")
    response = client.post("/cameras", json=NEW_CAMERA, headers=ADMIN)
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "admin_disabled"
    assert engine.added == {}
    assert client.get("/admin/check").json() == {"configured": False, "authorized": False}


def test_a_token_that_is_too_short_does_not_count(tmp_path):
    client, _ = make_client(tmp_path, admin_token="abc")
    assert client.post("/cameras", json=NEW_CAMERA, headers={"Authorization": "Bearer abc"}).status_code == 403


def test_camera_changes_need_the_right_token(tmp_path):
    client, engine = make_client(tmp_path)
    for headers in ({}, {"Authorization": "Bearer salah-sekali"}, {"Authorization": TOKEN}):
        response = client.post("/cameras", json=NEW_CAMERA, headers=headers)
        assert response.status_code == 401
        assert response.json()["detail"]["code"] == "bad_token"
    assert client.delete("/cameras/cam-5").status_code == 401
    assert client.put("/cameras/cam-5", json={}).status_code == 401
    assert client.post("/cameras/test", json={"url": SECRET_URL}).status_code == 401
    assert client.get("/admin/cameras").status_code == 401
    assert engine.added == {}
    assert client.get("/admin/check", headers=ADMIN).json() == {"configured": True, "authorized": True}
    assert client.get("/admin/check").json() == {"configured": True, "authorized": False}


def test_add_camera_passes_a_validated_config_to_the_engine(tmp_path):
    client, engine = make_client(tmp_path)
    response = client.post("/cameras", json=NEW_CAMERA, headers=ADMIN)
    assert response.status_code == 201
    assert response.json()["id"] == "cam-5"
    camera, url = engine.added["cam-5"]
    assert url == SECRET_URL
    assert (camera.description, camera.legacy_id, camera.roi, camera.process_fps) == (
        "Timbangan 3 masuk", 5, (0.1, 0.2, 0.9, 0.8), 6.0,
    )
    # The password never comes back, whatever the engine answered.
    assert "rahasia123" not in response.text


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"id": "kamera lima"}, "id"),
        ({"id": "../etc"}, "id"),
        ({"url": "http://192.168.1.210/video"}, "url"),
        ({"url": "file:///C:/Windows/win.ini"}, "url"),
        ({"url": "rtsp://"}, "url"),
        ({"roi": [0.5, 0.2, 0.4, 0.8]}, "roi"),
        ({"process_fps": 60}, "process_fps"),
        ({"legacy_id": 0}, "legacy_id"),
        ({"surprise": 1}, "surprise"),
    ],
)
def test_add_camera_rejects_bad_input(tmp_path, change, field):
    client, engine = make_client(tmp_path)
    response = client.post("/cameras", json={**NEW_CAMERA, **change}, headers=ADMIN)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == field
    assert engine.added == {}


def test_conflicts_carry_a_code_the_page_can_translate(tmp_path):
    client, _ = make_client(tmp_path)
    response = client.post("/cameras", json={**NEW_CAMERA, "id": "cam-1"}, headers=ADMIN)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "duplicate_id"
    # A camera of the config file cannot be changed through the API.
    response = client.put("/cameras/cam-1", json={"description": "x"}, headers=ADMIN)
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "config_camera"


def test_update_keeps_the_stored_url_when_none_is_given(tmp_path):
    client, engine = make_client(tmp_path)
    client.post("/cameras", json=NEW_CAMERA, headers=ADMIN)
    response = client.put("/cameras/cam-5", json={"description": "Timbangan 3 keluar", "url": "  "}, headers=ADMIN)
    assert response.status_code == 200
    camera, url = engine.added["cam-5"]
    assert camera.description == "Timbangan 3 keluar"
    assert url == SECRET_URL
    assert camera.legacy_id is None  # a PUT replaces every setting except the URL

    assert client.put("/cameras/cam-9", json={}, headers=ADMIN).status_code == 404


def test_remove_camera(tmp_path):
    client, engine = make_client(tmp_path)
    client.post("/cameras", json=NEW_CAMERA, headers=ADMIN)
    assert client.delete("/cameras/cam-5", headers=ADMIN).status_code == 204
    assert engine.added == {}
    response = client.delete("/cameras/cam-5", headers=ADMIN)
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "unknown_camera"


def test_connection_test_returns_stream_facts_and_a_picture(tmp_path):
    probe = FakeProbe()
    client, _ = make_client(tmp_path, probe=probe)
    response = client.post("/cameras/test", json={"url": SECRET_URL}, headers=ADMIN)
    assert response.status_code == 200
    body = response.json()
    assert (body["ok"], body["codec"], body["width"], body["height"], body["fps"]) == (True, "hevc", 2560, 1440, 20.0)
    assert body["snapshot"].startswith("data:image/jpeg;base64,")
    assert probe.urls == [SECRET_URL]
    assert "rahasia123" not in response.text


def test_connection_test_can_use_the_stored_url_of_a_camera(tmp_path):
    probe = FakeProbe()
    client, _ = make_client(tmp_path, probe=probe)
    assert client.post("/cameras/test", json={"camera_id": "cam-1"}, headers=ADMIN).json()["ok"] is True
    assert probe.urls == ["rtsp://admin:tersimpan@192.168.1.205:554/x"]
    assert client.post("/cameras/test", json={"camera_id": "cam-9"}, headers=ADMIN).status_code == 404
    assert client.post("/cameras/test", json={}, headers=ADMIN).status_code == 422
    assert client.post("/cameras/test", json={"url": "http://x/y"}, headers=ADMIN).status_code == 422


def test_connection_test_stops_after_repeated_rejected_logins(tmp_path):
    probe = FakeProbe()
    probe.result = ProbeResult("login_rejected", detail="HTTPUnauthorizedError")
    client, _ = make_client(tmp_path, probe=probe)
    for _ in range(3):
        response = client.post("/cameras/test", json={"url": SECRET_URL}, headers=ADMIN)
        assert response.status_code == 200
        assert response.json()["code"] == "login_rejected"
    # The fourth attempt would risk the camera locking the account: it is not even tried.
    response = client.post("/cameras/test", json={"url": SECRET_URL}, headers=ADMIN)
    assert response.status_code == 429
    assert response.json()["detail"]["code"] == "login_locked"
    assert int(response.headers["Retry-After"]) > 0
    assert len(probe.urls) == 3
    # Another camera is not affected.
    other = SECRET_URL.replace("192.168.1.210", "192.168.1.211")
    assert client.post("/cameras/test", json={"url": other}, headers=ADMIN).status_code == 200


def test_login_guard_forgets_old_failures_and_clears_on_success():
    guard = LoginGuard(max_failures=2, window_s=0.05)
    guard.record_rejection("cam")
    assert guard.retry_after("cam") == 0
    guard.record_rejection("cam")
    assert guard.retry_after("cam") > 0
    guard.clear("cam")
    assert guard.retry_after("cam") == 0
    guard.record_rejection("cam")
    guard.record_rejection("cam")
    time.sleep(0.1)
    assert guard.retry_after("cam") == 0


def test_admin_camera_list_hides_passwords(tmp_path):
    client, _ = make_client(tmp_path)
    response = client.get("/admin/cameras", headers=ADMIN)
    assert response.status_code == 200
    assert response.json()[0]["url"] == "rtsp://admin:***@192.168.1.205:554/x"


def test_snapshot_is_one_uncached_jpeg(tmp_path):
    client, engine = make_client(tmp_path)
    response = client.get("/cameras/cam-1/snapshot.jpg", params={"width": 700})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-store"
    assert response.content == b"\xff\xd8jpeg"
    assert engine.jpeg_calls == [640]  # rounded to the nearest of a few fixed widths
    assert client.get("/cameras/cam-9/snapshot.jpg").status_code == 404
    assert client.get("/cameras/cam-1/snapshot.jpg", params={"width": 5000}).status_code == 422
    assert client.get("/cameras/cam-9/video").status_code == 404


def test_snapshot_of_a_part_of_the_frame(tmp_path):
    client, engine = make_client(tmp_path)
    response = client.get("/cameras/cam-1/snapshot.jpg", params={"width": 1280, "crop": "0.25,0.3,0.75,0.8"})
    assert response.status_code == 200
    assert engine.jpeg_calls == [(1280, (0.25, 0.3, 0.75, 0.8))]
    for bad in ("0.5,0.3,0.25,0.8", "0,0,1", "0,0,1,1.5", "a,b,c,d", "nan,0,1,1"):
        assert client.get("/cameras/cam-1/snapshot.jpg", params={"crop": bad}).status_code == 422, bad


def test_browser_pages_are_served(tmp_path):
    client, _ = make_client(tmp_path)
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/ui"
    for path, marker in (("/ui", "dashboard.js"), ("/ui/cameras", "cameras.js")):
        page = client.get(path)
        assert page.status_code == 200
        assert page.headers["content-type"].startswith("text/html")
        assert marker in page.text
        # The pages must work on a PC without internet access.
        assert "https://" not in page.text
        assert "http://" not in page.text
    for name in ("app.css", "common.js", "dashboard.js", "cameras.js", "theme.js"):
        asset = client.get(f"/ui/static/{name}")
        assert asset.status_code == 200, name
        assert asset.headers["cache-control"] == "no-cache"


# -- region codes --------------------------------------------------------------------------


def test_region_list_is_open_and_changes_need_the_token(tmp_path):
    client, engine = make_client(tmp_path)
    listed = client.get("/regions")
    assert listed.status_code == 200
    assert listed.json() == [{"code": "BK", "name": "Sumatera Utara", "builtin": True, "seen": 12}]
    assert client.post("/regions", json={"code": "XK"}).status_code == 401
    assert client.delete("/regions/BK").status_code == 401
    assert client.post("/regions/restore").status_code == 401
    assert engine.regions == {"BK": "Sumatera Utara"}


def test_add_region_normalises_the_code(tmp_path):
    client, engine = make_client(tmp_path)
    response = client.post("/regions", json={"code": " xk ", "name": "  Kode uji  "}, headers=ADMIN)
    assert response.status_code == 201
    assert response.json() == {"code": "XK", "name": "Kode uji", "builtin": False, "seen": 0}
    assert engine.regions["XK"] == "Kode uji"


@pytest.mark.parametrize("code", ["", "B1", "ABC", "B K", "8", "Ä"])
def test_add_region_rejects_what_is_not_one_or_two_letters(tmp_path, code):
    client, engine = make_client(tmp_path)
    response = client.post("/regions", json={"code": code}, headers=ADMIN)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == "code"
    assert engine.regions == {"BK": "Sumatera Utara"}


def test_region_conflicts_and_removal(tmp_path):
    client, engine = make_client(tmp_path)
    duplicate = client.post("/regions", json={"code": "BK"}, headers=ADMIN)
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "duplicate_code"
    special = client.post("/regions", json={"code": "CD"}, headers=ADMIN)
    assert special.status_code == 409
    assert special.json()["detail"]["code"] == "special_code"

    assert client.delete("/regions/bk", headers=ADMIN).status_code == 204  # case does not matter
    assert engine.regions == {}
    missing = client.delete("/regions/BK", headers=ADMIN)
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "unknown_code"

    restored = client.post("/regions/restore", headers=ADMIN)
    assert restored.status_code == 200
    assert restored.json() == {"added": 1}


def test_region_page_and_its_scripts_are_served(tmp_path):
    client, _ = make_client(tmp_path)
    page = client.get("/ui/regions")
    assert page.status_code == 200
    assert "regions.js" in page.text
    assert "https://" not in page.text
    assert "http://" not in page.text
    for name in ("regions.js", "admin.js"):
        assert client.get(f"/ui/static/{name}").status_code == 200, name
    # Every page links to the two others.
    for path in ("/ui", "/ui/cameras", "/ui/regions"):
        text = client.get(path).text
        for link in ('href="/ui"', 'href="/ui/cameras"', 'href="/ui/regions"'):
            assert link in text, (path, link)
