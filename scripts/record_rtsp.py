"""Record camera streams to disk without re-encoding, for repeatable tests and benchmarks.

Reads camera URLs from environment variables named ALPR_CAM_<n>_URL and writes one
Matroska file per camera. The video packets are copied as they arrive (remux), so the
files hold exactly what the cameras sent.

Usage (from the project root):
    uv run --extra cpu --env-file .env python scripts/record_rtsp.py --seconds 600
    uv run --extra cpu --env-file .env python scripts/record_rtsp.py --seconds 60 --cameras 1 3
"""
from __future__ import annotations

import argparse
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import av

ENV_PATTERN = re.compile(r"^ALPR_CAM_(\w+)_URL$")


def mask(text: str) -> str:
    return re.sub(r"//([^:/@\s]+):[^@\s]+@", r"//\1:***@", text)


def camera_urls() -> dict[str, str]:
    found = {m.group(1).lower(): url for name, url in os.environ.items() if (m := ENV_PATTERN.match(name)) and url}
    return dict(sorted(found.items()))


def record(name: str, url: str, path: Path, seconds: float, results: dict[str, str]) -> None:
    packets = 0
    size = 0
    try:
        with av.open(url, options={"rtsp_transport": "tcp"}, timeout=(10, 10)) as source:
            video = source.streams.video[0]
            with av.open(str(path), "w", format="matroska") as target:
                out_stream = target.add_stream_from_template(video)
                started = time.monotonic()
                seen_keyframe = False
                for packet in source.demux(video):
                    if packet.dts is None:
                        continue
                    # Start the file on a keyframe so it can be decoded from the first packet.
                    seen_keyframe = seen_keyframe or packet.is_keyframe
                    if not seen_keyframe:
                        continue
                    size += packet.size
                    packets += 1
                    packet.stream = out_stream
                    target.mux(packet)
                    if time.monotonic() - started >= seconds:
                        break
        results[name] = f"ok: {packets} packets, {size / 1e6:.1f} MB -> {path}"
    except Exception as exc:  # noqa: BLE001 - report any failure, masked
        results[name] = mask(f"FAILED after {packets} packets: {type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--cameras", nargs="*", default=None, help="camera numbers to record (default: all)")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default data/samples/rec-<timestamp>)")
    args = parser.parse_args()

    urls = camera_urls()
    if args.cameras:
        urls = {name: url for name, url in urls.items() if name in set(args.cameras)}
    if not urls:
        raise SystemExit("No matching ALPR_CAM_<n>_URL variables. Run with: uv run --env-file .env python scripts/record_rtsp.py")
    out = args.out or Path("data/samples") / f"rec-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    results: dict[str, str] = {}
    threads = [
        threading.Thread(target=record, args=(name, url, out / f"cam-{name}.mkv", args.seconds, results), daemon=True)
        for name, url in urls.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for name in urls:
        print(f"cam-{name}: {results.get(name, 'no result')}", flush=True)


if __name__ == "__main__":
    main()
