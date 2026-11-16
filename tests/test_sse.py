"""The event stream over a real HTTP connection: publish in one thread, read in another."""
import http.client
import json
import socket
import threading
import time

import uvicorn

from alpr_id.api.app import create_app
from alpr_id.config import AppConfig, CameraConfig, StorageConfig
from alpr_id.pipeline.events import EventBus

EVENT = {"event_id": "abc123", "camera_id": "cam-1", "plate": "BK1234ABC", "plate_formatted": "BK 1234 ABC"}


class BusOnlyEngine:
    def __init__(self):
        self.bus = EventBus()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


def test_published_event_reaches_a_connected_client(tmp_path):
    config = AppConfig(
        cameras=(CameraConfig(id="cam-1", legacy_id=1, url_env="X"),),
        storage=StorageConfig(database=tmp_path / "db.sqlite3", media_dir=tmp_path / "media"),
    )
    engine = BusOnlyEngine()
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(config, engine=engine, manage_engine=False), host="127.0.0.1", port=port, log_config=None)
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        assert wait_until(lambda: server.started), "server did not start"

        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("GET", "/events/stream")
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("content-type").startswith("text/event-stream")
        assert wait_until(lambda: engine.bus.subscriber_count == 1), "client was not subscribed"

        # A camera thread would call this when a vehicle has been read.
        threading.Thread(target=engine.bus.publish, args=(EVENT,)).start()

        fields = {}
        while True:  # one message ends with an empty line
            line = response.readline().decode().rstrip("\r\n")
            if not line:
                if fields:
                    break
                continue
            if not line.startswith(":"):  # ':' lines are keep-alive comments
                name, _, value = line.partition(":")
                fields[name] = value.strip()
        assert fields["event"] == "plate"
        assert fields["id"] == "abc123"
        assert json.loads(fields["data"]) == EVENT

        connection.close()
        assert wait_until(lambda: engine.bus.subscriber_count == 0), "subscriber was not removed after disconnect"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
