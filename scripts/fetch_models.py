"""Download the public pretrained models used as the baseline into `models/`.

- OCR: fast-plate-ocr `cct-s-v2-global-model` (ONNX and its plate config), MIT licence.
- Comparison detector: open-image-models `yolo-v9-t-640-license-plate-end2end` (ONNX).

The main baseline detector is not downloaded: it is exported from the `best.pt` of the old
service with `training/detector/export_onnx.py`.

Usage (from the project root):
    uv run --extra cpu python scripts/fetch_models.py
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

FILES: dict[str, str] = {
    "models/ocr/cct_s_v2_global.onnx": "https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_s_v2_global.onnx",
    "models/ocr/cct_s_v2_global_plate_config.yaml": "https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_s_v2_global_plate_config.yaml",
    "models/detector/yolo-v9-t-640-license-plates-end2end.onnx": "https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-640-license-plates-end2end.onnx",
}


def download(url: str, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "model-fetch"})
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as handle:
        while chunk := response.read(1 << 16):
            handle.write(chunk)
    partial.replace(target)
    return target.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="download again even if the file exists")
    args = parser.parse_args()
    for relative, url in FILES.items():
        target = Path(relative)
        if target.is_file() and not args.force:
            print(f"exists   {relative} ({target.stat().st_size} bytes)")
            continue
        size = download(url, target)
        print(f"fetched  {relative} ({size} bytes)")


if __name__ == "__main__":
    main()
