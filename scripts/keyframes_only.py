"""Copy only the key frames of a long recording into small files, without decoding.

A day of four cameras is tens of gigabytes; scripts/scan_recording.py looks at key frames only,
and those are a fraction of it. This writes, next to a recording of scripts/record_long.py, a
folder of the same layout whose files hold just the key frame packets, with their time stamps
unchanged, so second s of a small file is second s of the original.

    python scripts/keyframes_only.py data/samples/rec-long-20261118 data/samples/rec-long-20261118-kf

Reading is sequential and nothing is decoded, so it can run on the inference PC next to the service.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import av


def copy_keyframes(source_path: Path, target_path: Path, every: float) -> tuple[int, int]:
    kept = size = 0
    last = None
    with av.open(str(source_path)) as source, av.open(str(target_path), "w", format="matroska") as target:
        video = source.streams.video[0]
        out_stream = target.add_stream_from_template(video)
        for packet in source.demux(video):
            if packet.dts is None or not packet.is_keyframe:
                continue
            moment = float(packet.pts * video.time_base)
            if last is not None and moment - last < every:
                continue
            last = moment
            size += packet.size
            kept += 1
            packet.stream = out_stream
            target.mux(packet)
    return kept, size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--every", type=float, default=0.0, help="least seconds between kept key frames (default: keep all)")
    args = parser.parse_args()

    total = 0
    for folder in sorted(path for path in args.source.iterdir() if path.is_dir()):
        (args.target / folder.name).mkdir(parents=True, exist_ok=True)
        frames = size = 0
        for path in sorted(folder.glob("*.mkv")):
            try:
                kept, written = copy_keyframes(path, args.target / folder.name / path.name, args.every)
            except av.error.FFmpegError as exc:
                print(f"{folder.name}/{path.name}: skipped ({type(exc).__name__})", flush=True)
                continue
            frames += kept
            size += written
        total += size
        print(f"{folder.name}: {frames} key frames, {size / 1e9:.2f} GB", flush=True)
    print(f"total {total / 1e9:.2f} GB -> {args.target}")


if __name__ == "__main__":
    main()
