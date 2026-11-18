"""Record the camera streams for hours, without re-encoding, in files of a fixed length.

For questions a few minutes of video cannot answer, such as from which distance a plate is
found and read: record a day and a night on the inference PC, then run the stretches of
interest through the pipeline (scripts/compare_focus.py) next to the events the service stored.

    python scripts/record_long.py --env-file .env --out data/samples/rec-long-20261118 --until 2026-11-19T18:00

One folder per camera, one Matroska file per --segment-minutes, named after the server clock at
its first packet (20261118-160501.mkv): second s of a file is that time plus s. `record.log` in
--out says when each file started and ended. Between two files of a camera the stream is opened
again, which costs a few seconds of video.

It stops by itself at --until, when the disk has less than --min-free-gb left, or when a file
named STOP appears in --out. A camera that refuses the login three times in a row is left alone
for the rest of the run, so a wrong password cannot lock the account. Other failures are retried
with a growing pause. Camera addresses come from ALPR_CAM_<n>_URL, in the environment or in
--env-file, and are never written to the log with their password.
"""
from __future__ import annotations

import argparse
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import av
from record_rtsp import camera_urls, mask

MAX_REFUSED_LOGINS = 3
RETRY_FROM_S, RETRY_UP_TO_S = 15.0, 600.0

stop = threading.Event()
log_lock = threading.Lock()


def read_env_file(path: Path) -> None:
    """KEY=VALUE lines into the environment, for a run without `uv run --env-file`."""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def say(log: Path, text: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {mask(text)}"
    with log_lock, log.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def record_segment(url: str, folder: Path, seconds: float) -> tuple[Path | None, int, int]:
    """One file: from the first key frame until `seconds` have passed. Returns its path, packets and bytes."""
    packets = size = 0
    path = None
    with av.open(url, options={"rtsp_transport": "tcp"}, timeout=(10, 10)) as source:
        video = source.streams.video[0]
        target = out_stream = None
        started = time.monotonic()
        try:
            for packet in source.demux(video):
                if packet.dts is None:
                    continue
                if target is None:
                    if not packet.is_keyframe:
                        continue  # a file that starts between key frames cannot be decoded from its start
                    path = folder / f"{datetime.now():%Y%m%d-%H%M%S}.mkv"
                    target = av.open(str(path), "w", format="matroska")
                    out_stream = target.add_stream_from_template(video)
                    started = time.monotonic()
                size += packet.size
                packets += 1
                packet.stream = out_stream
                target.mux(packet)
                if time.monotonic() - started >= seconds or stop.is_set():
                    break
        finally:
            if target is not None:
                target.close()
    return path, packets, size


def run_camera(name: str, url: str, out: Path, until: float, segment_s: float, min_free_gb: float, log: Path) -> None:
    folder = out / f"cam-{name}"
    folder.mkdir(parents=True, exist_ok=True)
    refused, pause = 0, RETRY_FROM_S
    while not stop.is_set() and time.time() < until:
        free_gb = shutil.disk_usage(out).free / 1e9
        if free_gb < min_free_gb:
            say(log, f"cam-{name}: {free_gb:.0f} GB left on the disk, below --min-free-gb; stopping everything")
            stop.set()
            break
        try:
            path, packets, size = record_segment(url, folder, min(segment_s, until - time.time()))
        except Exception as exc:  # noqa: BLE001 - any failure is logged, masked, and retried
            kind = type(exc).__name__
            say(log, f"cam-{name}: FAILED {kind}: {exc}")
            if "Unauthorized" in kind or "401" in str(exc):
                refused += 1
                if refused >= MAX_REFUSED_LOGINS:
                    say(log, f"cam-{name}: login refused {refused} times in a row; this camera is not tried again")
                    break
            stop.wait(pause)
            pause = min(pause * 2, RETRY_UP_TO_S)
            continue
        if packets:
            refused, pause = 0, RETRY_FROM_S
            say(log, f"cam-{name}: {path.name} closed, {packets} packets, {size / 1e6:.0f} MB")
        else:
            say(log, f"cam-{name}: no video arrived")
            stop.wait(pause)
            pause = min(pause * 2, RETRY_UP_TO_S)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--until", type=datetime.fromisoformat, required=True, help="server time at which recording ends")
    parser.add_argument("--segment-minutes", type=float, default=30.0)
    parser.add_argument("--min-free-gb", type=float, default=100.0, help="stop when the disk of --out has less than this left")
    parser.add_argument("--cameras", nargs="*", default=None, help="camera numbers to record (default: all)")
    parser.add_argument("--env-file", type=Path, default=None)
    args = parser.parse_args()

    if args.env_file:
        read_env_file(args.env_file)
    urls = camera_urls()
    if args.cameras:
        urls = {name: url for name, url in urls.items() if name in set(args.cameras)}
    if not urls:
        raise SystemExit("No matching ALPR_CAM_<n>_URL variables; give --env-file .env")
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / "record.log"
    say(log, f"recording cameras {', '.join(urls)} until {args.until:%Y-%m-%d %H:%M}, files of {args.segment_minutes:g} minutes, "
             f"stop below {args.min_free_gb:g} GB free; pid {os.getpid()}")

    threads = [
        threading.Thread(target=run_camera, daemon=True,
                         args=(name, url, args.out, args.until.timestamp(), args.segment_minutes * 60, args.min_free_gb, log))
        for name, url in urls.items()
    ]
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads):
        if (args.out / "STOP").exists():
            say(log, "STOP file found; closing the files")
            stop.set()
        time.sleep(2)
    say(log, "recording ended")


if __name__ == "__main__":
    main()
