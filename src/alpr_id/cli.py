"""Command line entry point: serve, recognize, benchmark."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
from pathlib import Path

from alpr_id import __version__


def _disable_console_quick_edit() -> None:
    """Stop a mouse click in the console window from freezing the service.

    With QuickEdit on, clicking inside the window pauses every write to it. The log handler
    then blocks while holding its lock and all camera threads stop at their next log line.
    """
    if sys.platform != "win32":
        return
    # A safeguard only: whatever goes wrong here is never worth failing startup for.
    with contextlib.suppress(Exception):
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            quick_edit, extended_flags = 0x0040, 0x0080
            kernel32.SetConsoleMode(handle, (mode.value & ~quick_edit) | extended_flags)


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    from alpr_id.api.app import create_app
    from alpr_id.config import load_config
    from alpr_id.logging_setup import configure_logging, get_logger

    config = load_config(args.config)
    configure_logging()
    host = args.host or os.environ.get("ALPR_HOST", "0.0.0.0")
    port = args.port or int(os.environ.get("ALPR_PORT", "5000"))
    get_logger("alpr_id").info("server_starting", host=host, port=port, version=__version__)
    # One worker only: every worker process would open its own camera streams.
    _disable_console_quick_edit()
    # Without the shutdown timeout, Ctrl+C waits forever for open video or event streams.
    uvicorn.run(create_app(config), host=host, port=port, workers=1, log_config=None, timeout_graceful_shutdown=5)


def _recognize(args: argparse.Namespace) -> None:
    import cv2
    import numpy as np

    from alpr_id.config import load_config
    from alpr_id.logging_setup import configure_logging
    from alpr_id.pipeline.engine import Engine, load_models

    config = load_config(args.config)
    configure_logging(log_dir=None, level="WARNING")
    image = cv2.imdecode(np.frombuffer(Path(args.image).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        sys.exit(f"Not a readable image: {args.image}")
    engine = Engine(config)
    engine.detector, engine.ocr = load_models(config)
    print(json.dumps(engine.recognize(image), indent=2))


def _benchmark(args: argparse.Namespace) -> None:
    from alpr_id.bench.report import benchmark_live, benchmark_recordings, save_report, to_markdown
    from alpr_id.config import load_config
    from alpr_id.logging_setup import configure_logging

    config = load_config(args.config)
    configure_logging(log_dir=None, level="WARNING")
    if args.live:
        report = benchmark_live(config, args.live)
    else:
        files: dict[str, Path] = {}
        for item in args.recordings:
            path = Path(item)
            candidates = sorted(path.glob("*.mkv")) if path.is_dir() else [path]
            for candidate in candidates:
                # Files written by scripts/record_rtsp.py are named cam-<n>.mkv.
                match = re.fullmatch(r"cam-(\d+)", candidate.stem)
                camera = config.camera_by_legacy_id(int(match.group(1))) if match else config.camera(candidate.stem)
                if camera is None:
                    sys.exit(f"Cannot tell which camera {candidate} belongs to (expected cam-<n>.mkv)")
                files[camera.id] = candidate
        if not files:
            sys.exit("No recordings found")
        crops = Path(args.out) / "crops" if args.save_crops else None
        report = benchmark_recordings(config, files, start=args.start, end=args.end, crops_dir=crops)
    json_path, md_path = save_report(report, args.out)
    print(to_markdown(report))
    print(f"saved {json_path} and {md_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="alpr-id", description="Indonesian licence plate reader for RTSP CCTV")
    parser.add_argument("--version", action="version", version=f"alpr-id {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the cameras and the REST API")
    serve.add_argument("--config", default=None, help="path to the YAML config (default config/app.yaml)")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.set_defaults(func=_serve)

    recognize = sub.add_parser("recognize", help="read the plates in one image file")
    recognize.add_argument("image")
    recognize.add_argument("--config", default=None)
    recognize.set_defaults(func=_recognize)

    benchmark = sub.add_parser("benchmark", help="measure latency per stage")
    benchmark.add_argument("recordings", nargs="*", help="recording files or a folder of cam-<n>.mkv files")
    benchmark.add_argument("--live", type=float, default=None, metavar="SECONDS", help="run on the real cameras instead")
    benchmark.add_argument("--start", type=float, default=0.0, help="start position in the recordings, seconds")
    benchmark.add_argument("--end", type=float, default=None, help="end position in the recordings, seconds")
    benchmark.add_argument("--save-crops", action="store_true", help="save the plate crop of every event for checking by eye")
    benchmark.add_argument("--out", default="data/benchmarks")
    benchmark.add_argument("--config", default=None)
    benchmark.set_defaults(func=_benchmark)

    args = parser.parse_args(argv)
    args.func(args)
