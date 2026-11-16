"""Measure what each camera stream really delivers: codec, resolution, fps, bitrate.

Reads camera URLs from environment variables named ALPR_CAM_<n>_URL, decodes each
stream for a few seconds and saves one frame per stream. Credentials are masked in
everything it prints or writes.

Usage (from the project root):
    uv run --env-file .env python scripts/probe_rtsp.py
    uv run --env-file .env python scripts/probe_rtsp.py --sub --seconds 10
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import av
import cv2

ENV_PATTERN = re.compile(r"^ALPR_CAM_(\w+)_URL$")


def mask(text: str) -> str:
    """Hide the password of any rtsp://user:password@host URL inside text."""
    return re.sub(r"//([^:/@\s]+):[^@\s]+@", r"//\1:***@", text)


def camera_urls() -> dict[str, str]:
    found = {m.group(1).lower(): url for name, url in os.environ.items() if (m := ENV_PATTERN.match(name)) and url}
    return dict(sorted(found.items()))


def probe(url: str, seconds: float, sample_path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"url": mask(url)}
    started = time.monotonic()
    try:
        container = av.open(url, options={"rtsp_transport": "tcp"}, timeout=10)
    except Exception as exc:  # noqa: BLE001 - report any open failure, masked
        info["error"] = mask(f"{type(exc).__name__}: {exc}")
        return info
    try:
        stream = container.streams.video[0]
        ctx = stream.codec_context
        info.update(
            codec=ctx.name,
            profile=ctx.profile,
            width=ctx.width,
            height=ctx.height,
            nominal_fps=float(stream.average_rate) if stream.average_rate else None,
            open_s=round(time.monotonic() - started, 2),
        )
        frames = keyframes = size = 0
        first_pts = last_pts = None
        last_frame = None
        read_started = time.monotonic()
        for packet in container.demux(stream):
            if packet.size:
                size += packet.size
                keyframes += int(packet.is_keyframe)
            for frame in packet.decode():
                frames += 1
                if frame.pts is not None:
                    first_pts = frame.pts if first_pts is None else first_pts
                    last_pts = frame.pts
                last_frame = frame
            if time.monotonic() - read_started >= seconds:
                break
        wall = time.monotonic() - read_started
        info.update(
            frames_decoded=frames,
            read_s=round(wall, 2),
            fps_received=round(frames / wall, 2) if wall else None,
            keyframes=keyframes,
            kbps=round(size * 8 / wall / 1000) if wall else None,
        )
        if frames > 1 and first_pts is not None and last_pts is not None and stream.time_base:
            span = float((last_pts - first_pts) * stream.time_base)
            # Lower than nominal_fps means frames were lost on the way.
            info["fps_by_timestamp"] = round((frames - 1) / span, 2) if span > 0 else None
        if last_frame is not None:
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(sample_path), last_frame.to_ndarray(format="bgr24"))
            info["sample"] = str(sample_path)
    except Exception as exc:  # noqa: BLE001
        info["error"] = mask(f"{type(exc).__name__}: {exc}")
    finally:
        container.close()
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=5.0, help="how long to read each stream")
    parser.add_argument("--sub", action="store_true", help="also probe the Hikvision sub stream (channel x02)")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default data/samples/probe-<timestamp>)")
    args = parser.parse_args()

    urls = camera_urls()
    if not urls:
        raise SystemExit("No ALPR_CAM_<n>_URL variables found. Run with: uv run --env-file .env python scripts/probe_rtsp.py")
    out = args.out or Path("data/samples") / f"probe-{datetime.now():%Y%m%d-%H%M%S}"

    results = []
    for name, url in urls.items():
        targets = [("main", url)]
        if args.sub:
            targets.append(("sub", re.sub(r"(/Channels/\d)01\b", r"\g<1>02", url, flags=re.IGNORECASE)))
        for kind, target in targets:
            result = {"camera": name, "stream": kind, **probe(target, args.seconds, out / f"cam-{name}-{kind}.jpg")}
            results.append(result)
            print(json.dumps(result), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
